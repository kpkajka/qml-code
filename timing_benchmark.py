#!/usr/bin/env python3
"""
timing_benchmark.py  --  per-event forward- and backward-pass timing per architecture.

Builds each circuit fresh (untrained weights; timing is independent of parameter values), 
times a single-event forward pass (inference) and a single-event
backward pass (loss + gradients), after warmup runs that exclude one-time TensorFlow
graph compilation. Uses the runner's own forward path (collect_batch_predictions ->
binary_cross_entropy_from_logits).

    python timing_benchmark.py --runner sf_classifier_entanglement_runner_new.py \
        --path HToCC_vs_ZJets_flat.h5 --out timing.csv
"""

import argparse, csv, importlib.util, time
import numpy as np
import tensorflow as tf


def load_runner(path):
    spec = importlib.util.spec_from_file_location("runner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# (display name, arch, ck_topology, use_self_kerr)
ARCHES = [
    ("Int-S-Int",    "new_entangled",     "chain", False),
    ("Int-K-Int",    "new_entangled_k",   "chain", False),
    ("Int-K0-Int",   "new_entangled_k0",  "chain", False),
    ("Int-CK-Int",   "new_entangled_kck", "chain", False),   # ck-only
    ("Int-K-CK-Int", "new_entangled_kck", "chain", True),    # with self-kerr
]


def time_arch(R, name, arch, ck_topology, use_self_kerr, X1, y1,
              modes, layers, cutoff, encoding_mode, local_gates, observable_readout,
              warmup, reps):
    
    if arch in {"fixed_ring", "trainable_ring", "fixed_alltoall", "trainable_alltoall",
                "fixed_chain", "trainable_chain", "fixed_star", "trainable_star"}:
        pairs = R.TOPOS[arch.split("_")[1]](modes)
    else:
        pairs = []

    seed = 0
    enc_vars = R.init_encoder_vars(modes, seed, encoding_mode)
    arch_vars = R.init_arch_vars(arch, pairs, modes, layers, seed, ck_topology, use_self_kerr)
    local_vars = R.init_local_vars(modes, layers, local_gates, seed)
    clf_head = R.init_classifier_head_for_readout(modes, seed, observable_readout)

    # trainable variables (for the gradient in the backward pass)
    train_vars = []
    for dct in (enc_vars, arch_vars, local_vars, clf_head):
        for v in dct.values():
            if isinstance(v, tf.Variable):
                train_vars.append(v)

    yb = tf.constant(y1.astype(np.float32))

    def logits_of(Xb):
        # fresh engine per call (matches run_circuit's run+reset discipline and
        # avoids stale engine state across forward/backward calls)
        engine = R.sf.Engine("tf", backend_options={"cutoff_dim": int(cutoff)})
        return R.collect_batch_predictions(
            Xb, modes, arch, pairs, layers, cutoff, enc_vars, arch_vars,
            local_vars, local_gates, clf_head, engine, encoding_mode,
            observable_readout, True)[0]

    def forward():
        return logits_of(X1)

    def backward():
        with tf.GradientTape() as tape:
            logits = logits_of(X1)
            loss = R.binary_cross_entropy_from_logits(logits, yb)
        return tape.gradient(loss, train_vars)

    # warmup (excluded)
    try:
        for _ in range(warmup): forward()
        for _ in range(warmup): backward()
    except Exception as e:
        print(f"  [{name}] warmup error: {type(e).__name__}: {e}")
        return None

    ft, bt = [], []
    for _ in range(reps):
        t0 = time.perf_counter(); forward(); ft.append(time.perf_counter() - t0)
    for _ in range(reps):
        t0 = time.perf_counter(); backward(); bt.append(time.perf_counter() - t0)

    ft = np.array(ft) * 1e3; bt = np.array(bt) * 1e3   # ms

    def med_iqr(a):
        med = np.median(a)
        iqr = np.percentile(a, 75) - np.percentile(a, 25)   # interquartile range
        return med, iqr
    fmed, fiqr = med_iqr(ft)
    bmed, biqr = med_iqr(bt)
    return fmed, fiqr, bmed, biqr, len(train_vars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--modes", type=int, default=6)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--cutoff", type=int, default=6)
    ap.add_argument("--encoding_mode", default="trainable_per_mode")
    ap.add_argument("--local_gates", default="none")
    ap.add_argument("--observable_readout", default="mean_photon")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="timing.csv")
    args = ap.parse_args()

    R = load_runner(args.runner)

    import h5py
    with h5py.File(args.path, "r") as f:
        #const = f["jetConstituentsList"][:2].astype(np.float32)
        const = f["jetConstituentsList"][:1000].astype(np.float32)
        names = [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
                 for x in f["jetFeatureNames"][()]]
        #jet_pt = f["jetFeatures"][:2, names.index("jet_pt")].astype(np.float32)
        jet_pt = f["jetFeatures"][:1000, names.index("jet_pt")].astype(np.float32)
    X1 = R.build_topk_features_for_indices(const, jet_pt, [2], args.modes)
    y1 = np.array([1.0], dtype=np.float32)

    rows = []
    print(f"{'architecture':<16}{'fwd ms':>10}{'bwd ms':>10}{'bwd/fwd':>10}")
    for name, arch, cktop, selfk in ARCHES:
        res = time_arch(R, name, arch, cktop, selfk, X1, y1,
                        args.modes, args.layers, args.cutoff,
                        args.encoding_mode, args.local_gates, args.observable_readout,
                        args.warmup, args.reps)
        if res is None:
            print(f"{name:<16}{'(failed)':>30}"); continue
        fmean, fstd, bmean, bstd, npar = res
        ratio = bmean / fmean if fmean > 0 else float("nan")
        rows.append({"architecture": name, "forward_ms": round(fmean, 3),
                     "forward_std_ms": round(fstd, 3), "backward_ms": round(bmean, 3),
                     "backward_std_ms": round(bstd, 3),
                     "backward_over_forward": round(ratio, 2), "n_trainable": npar})
        print(f"{name:<16}{fmean:>8.2f}  {bmean:>8.2f}  {ratio:>8.2f}")

    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["architecture", "forward_ms", "forward_std_ms",
                                           "backward_ms", "backward_std_ms",
                                           "backward_over_forward", "n_trainable"])
        wr.writeheader(); wr.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
