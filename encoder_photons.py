#!/usr/bin/env python3
"""
For each architecture, loads each seed's trained encoder weights, runs only the
encoder (D, S, R on the vacuum) on a jet sample, reads the mean photon number
<n> of every mode, and writes all values to a CSV. 

CSV columns: architecture, seed, jet, mode, nbar

    python encoder_photons_compute.py --runner sf_classifier_entanglement_runner_new.py \
        --root classifier/runs --path data/HToCC_vs_ZJets_flat.h5 \
        --out encoder_nbar.csv
"""

import argparse, glob, os, csv, importlib.util, re
import numpy as np
import h5py


def load_runner(path):
    spec = importlib.util.spec_from_file_location("runner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def seed_of(path):
    m = re.search(r"seed(\d+)", path)
    return int(m.group(1)) if m else None


DEFAULT_ARCHES = [
    ("Int-S-Int",    "int_s_int/m6/L1/new_entangled/lg_none/cutoff_6",              "trainable_per_mode"),
    ("Int-K-Int",    "int_k_int/m6/L1/new_entangled_k/lg_none/cutoff_6",            "trainable_per_mode"),
    ("Int-K0-Int",   "int_k0_int/m6/L1/new_entangled_k0/lg_none/cutoff_6",          "trainable_per_mode"),
    ("Int-CK-Int",   "int_kck_int/m6/L1/new_entangled_kck/lg_none/cutoff_6/ck_chain_ckonly",   "trainable_per_mode"),
    ("Int-K-CK-Int", "int_kck_int/m6/L1/new_entangled_kck/lg_none/cutoff_6/ck_chain_selfkerr", "trainable_per_mode"),
]


def overwrite_encoder(enc_vars, payload):
    for key, var in enc_vars.items():
        pk = f"encoder__{key}"
        if pk in payload:
            var.assign(payload[pk].astype(np.float32))


def encoder_fock_probs_for_jets(R, X, enc_vars, encoding_mode, modes, cutoff):
    """Run the encoder on each jet, return the per-mode photon-number
    distribution P(n). Shape [n_jets, modes, cutoff]. No distributional
    assumption -- these are the exact Fock probabilities from the state."""
    import tensorflow as tf
    out = np.zeros((len(X), modes, cutoff), dtype=np.float64)
    for j in range(len(X)):
        prog = R.sf.Program(modes)
        with prog.context as q:
            R.encode_input(q, X[j], modes, enc_vars, encoding_mode)   # encoder ONLY
        eng = R.sf.Engine("tf", backend_options={"cutoff_dim": int(cutoff)})
        probs = eng.run(prog).state.all_fock_probs()
        for m in range(modes):
            axes = [ax for ax in range(modes) if ax != m]
            marginal = tf.reduce_sum(probs, axis=axes)   # P(n) for mode m
            out[j, m, :] = np.asarray(marginal, dtype=np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--modes", type=int, default=6)
    ap.add_argument("--cutoff", type=int, default=6)
    ap.add_argument("--n_jets", type=int, default=100)
    ap.add_argument("--max_seeds", type=int, default=10)
    ap.add_argument("--arches", nargs="+", default=None)
    ap.add_argument("--out", default="encoder_nbar.csv")
    args = ap.parse_args()

    R = load_runner(args.runner)
    arches = ([tuple(e.split(":", 2)) for e in args.arches] if args.arches else DEFAULT_ARCHES)

    with h5py.File(args.path, "r") as f:
        const = f["jetConstituentsList"][:args.n_jets].astype(np.float32)
        names = [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
                 for x in f["jetFeatureNames"][()]]
        jet_pt = f["jetFeatures"][:args.n_jets, names.index("jet_pt")].astype(np.float32)
    idx = list(range(args.n_jets))
    X = R.build_topk_features_for_indices(const, jet_pt, idx, args.modes)

    fcsv = open(args.out, "w", newline="")
    writer = csv.DictWriter(fcsv, fieldnames=["architecture", "seed", "jet", "mode", "p0", "p1", "p_ge2", "nbar"])
    writer.writeheader()

    print(f"{'architecture':<16}{'seeds':>7}{'median <n>':>12}", flush=True)
    for entry in arches:
        name, rel, enc_mode = entry
        arch_dir = os.path.join(args.root, rel)
        wfs = sorted(glob.glob(f"{arch_dir}/**/seed*/trained_model_weights.npz", recursive=True))
        wfs = wfs[:args.max_seeds]
        if not wfs:
            print(f"{name:<16}{'(no seeds)':>19}", flush=True); continue

        vals_for_median = []
        for wf in wfs:
            payload = dict(np.load(wf, allow_pickle=True))
            s = seed_of(wf)
            enc_vars = R.init_encoder_vars(args.modes, s, enc_mode)
            overwrite_encoder(enc_vars, payload)
            pns = encoder_fock_probs_for_jets(R, X, enc_vars, enc_mode, args.modes, args.cutoff)
            ncols = np.arange(pns.shape[2])                       # [0,1,2,...,cutoff-1]
            for j in range(pns.shape[0]):
                for m in range(pns.shape[1]):
                    pn = pns[j, m]
                    p0 = float(pn[0])
                    p1 = float(pn[1]) if pn.size > 1 else 0.0
                    p_ge2 = float(pn[2:].sum()) if pn.size > 2 else 0.0
                    nbar = float((ncols * pn).sum())              # <n> = sum n P(n)
                    writer.writerow({"architecture": name, "seed": s, "jet": j, "mode": m,
                                     "p0": p0, "p1": p1, "p_ge2": p_ge2, "nbar": nbar})
                    vals_for_median.append(nbar)
            fcsv.flush()
        med = float(np.median(vals_for_median))
        print(f"{name:<16}{len(wfs):>7}{med:>12.3f}", flush=True)

    fcsv.close()
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
