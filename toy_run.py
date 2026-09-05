#!/usr/bin/env python3
"""
Gaussian (Int-S-Int) vs Kerr (Int-K-Int) CV circuit on 2D toy datasets, photon-
number readout. Writes one JSON per seed into a folder tree.

Folder layout:
  <outbase>/<dataset>/enc{enc_scale}_L{layers}_cut{cutoff}_ep{epochs}/<arch>/seed{N}/final_summary.json

Each final_summary.json records: config, final AUC, full per-epoch loss+AUC
history, trained middle-gate magnitudes, mean photon number entering the middle 
gate, and a leakage estimate (Fock mass at the cutoff edge).

    python toy_run.py --datasets rings --enc_scale 2.2 --seeds 8 --epochs 150 \
        --layers 2 --cutoff 20 --outbase results
"""

import argparse, json, time
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.datasets import make_circles
from sklearn.metrics import roc_auc_score
import strawberryfields as sf
from strawberryfields import ops


# datasets
def make_xor(n, seed=0, noise=0.12):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    return X + rng.normal(0, noise, X.shape), y

def make_rings(n, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.uniform(0, 3, n); th = rng.uniform(0, 2*np.pi, n)
    X = np.column_stack([r*np.cos(th), r*np.sin(th)]) / 3.0
    y = (np.floor(r) % 2).astype(int)
    return X, y

def load_dataset(name, n=300, seed=0):
    if name == "rings":   return make_rings(n, seed)
    if name == "circles": return make_circles(n_samples=n, noise=0.08, factor=0.4, random_state=seed)
    if name == "xor":     return make_xor(n, seed)
    raise ValueError(name)


# circuit
def init_weights(seed, layers, kappa_std=0.5):
    g = tf.random.Generator.from_seed(seed)
    v = lambda shape=(), s=0.1: tf.Variable(g.normal(shape, stddev=s))
    w = {"w": v([2]), "b": v()}
    for L in range(layers):
        w[f"t1_{L}"] = v(); w[f"p1_{L}"] = v()
        w[f"t2_{L}"] = v(); w[f"p2_{L}"] = v()
        w[f"m0_{L}"] = v(s=kappa_std); w[f"m1_{L}"] = v(s=kappa_std)
    return w

def build_program(layers, arch):
    prog = sf.Program(2)
    names = ["x0", "x1"]
    for L in range(layers):
        names += [f"t1_{L}", f"p1_{L}", f"t2_{L}", f"p2_{L}", f"m0_{L}", f"m1_{L}"]
    P = prog.params(*names); sym = dict(zip(names, P))
    with prog.context as q:
        ops.Dgate(sym["x0"]) | q[0]; ops.Dgate(sym["x1"]) | q[1]
        for L in range(layers):
            ops.BSgate(sym[f"t1_{L}"], sym[f"p1_{L}"]) | (q[0], q[1])
            if arch == "gaussian":
                ops.Sgate(sym[f"m0_{L}"]) | q[0]; ops.Sgate(sym[f"m1_{L}"]) | q[1]
            else:
                ops.Kgate(sym[f"m0_{L}"]) | q[0]; ops.Kgate(sym[f"m1_{L}"]) | q[1]
            ops.BSgate(sym[f"t2_{L}"], sym[f"p2_{L}"]) | (q[0], q[1])
    return prog

def run_state(X, w, arch, layers, cutoff, enc_scale):
    B = X.shape[0]
    eng = sf.Engine("tf", backend_options={"cutoff_dim": cutoff, "batch_size": B})
    prog = build_program(layers, arch)
    args = {"x0": tf.constant(enc_scale*X[:,0], tf.float32),
            "x1": tf.constant(enc_scale*X[:,1], tf.float32)}
    for L in range(layers):
        for k in (f"t1_{L}", f"p1_{L}", f"t2_{L}", f"p2_{L}", f"m0_{L}", f"m1_{L}"):
            args[k] = w[k]
    return eng.run(prog, args=args).state

def logits_from_state(st, w):
    n0 = tf.math.real(st.mean_photon(0)[0]); n1 = tf.math.real(st.mean_photon(1)[0])
    feats = tf.stack([n0, n1], axis=1)
    return tf.linalg.matvec(feats, w["w"]) + w["b"]

def run_circuit(X, w, arch, layers, cutoff, enc_scale):
    return logits_from_state(run_state(X, w, arch, layers, cutoff, enc_scale), w)


def nbar_entering_first_mid(X, w, cutoff, enc_scale):
    B = X.shape[0]
    eng = sf.Engine("tf", backend_options={"cutoff_dim": cutoff, "batch_size": B})
    prog = sf.Program(2)
    x0, x1, t1, p1 = prog.params("x0","x1","t1","p1")
    with prog.context as q:
        ops.Dgate(x0) | q[0]; ops.Dgate(x1) | q[1]
        ops.BSgate(t1, p1) | (q[0], q[1])
    args = {"x0": tf.constant(enc_scale*X[:,0], tf.float32),
            "x1": tf.constant(enc_scale*X[:,1], tf.float32),
            "t1": w["t1_0"], "p1": w["p1_0"]}
    st = eng.run(prog, args=args).state
    return float(tf.reduce_mean(tf.math.real(st.mean_photon(0)[0] + st.mean_photon(1)[0])))

def leakage_at_cutoff(X, w, arch, layers, cutoff, enc_scale, sample=120):
    """Mean probability mass in the TOP Fock level of either mode -- a proxy for
    truncation error. If this is non-negligible, the cutoff is too low and the
    circuit's behaviour (esp. a 'Gaussian' one exceeding the quadratic ceiling)
    may be a truncation artifact rather than real."""
    Xs = X[:sample]
    st = run_state(Xs, w, arch, layers, cutoff, enc_scale)
    p = st.all_fock_probs()                        # (B, cutoff, cutoff)
    p = tf.math.real(p).numpy()
    top = cutoff - 1
    edge0 = p[:, top, :].sum(axis=1)               # mass with mode0 at top level
    edge1 = p[:, :, top].sum(axis=1)               # mass with mode1 at top level
    edge = np.maximum(edge0, edge1)
    return {"edge_prob_mean": float(edge.mean()),
            "edge_prob_max": float(edge.max()),
            "frac_gt_1pct": float((edge > 0.01).mean())}


def train_seed(X, y, arch, seed, epochs, layers, cutoff, enc_scale, lr=0.08):
    w = init_weights(seed, layers)
    opt = tf.keras.optimizers.Adam(lr)
    yt = tf.constant(y, tf.float32)
    varlist = list(w.values())
    hist = {"epoch": [], "loss": [], "auc": []}
    for ep in range(epochs):
        with tf.GradientTape() as tape:
            logits = run_circuit(X, w, arch, layers, cutoff, enc_scale)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=yt, logits=logits))
        opt.apply_gradients(zip(tape.gradient(loss, varlist), varlist))
        ep_logits = run_circuit(X, w, arch, layers, cutoff, enc_scale).numpy()
        hist["epoch"].append(ep + 1)
        hist["loss"].append(float(loss))
        hist["auc"].append(float(roc_auc_score(y, ep_logits)))

    # trained middle-gate magnitudes (squeeze for gaussian, kappa for kerr)
    mid_mag = [[float(tf.abs(w[f"m0_{L}"])), float(tf.abs(w[f"m1_{L}"]))] for L in range(layers)]
    nbar = nbar_entering_first_mid(X, w, cutoff, enc_scale)
    leak = leakage_at_cutoff(X, w, arch, layers, cutoff, enc_scale)

    # "still rising" flag: AUC change over the last 20 epochs
    auc = hist["auc"]
    trend = auc[-1] - auc[-20] if len(auc) >= 20 else auc[-1] - auc[0]

    return {
        "seed": seed, "arch": arch,
        "final_auc": auc[-1],
        "last20_auc_change": trend,
        "still_rising": bool(trend > 0.01),
        "mid_gate": "squeeze" if arch == "gaussian" else "kerr",
        "mid_magnitude_per_layer": mid_mag,   # |squeeze r| or |kappa|
        "nbar_entering_first_mid": nbar,
        "leakage": leak,                      # truncation diagnostics
        "history": hist,
    }, {k: v.numpy() for k, v in w.items()}    # trained weights, for decision plots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["rings"])
    ap.add_argument("--archs", nargs="+", default=["gaussian", "kerr"])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--cutoff", type=int, default=12)
    ap.add_argument("--enc_scale", type=float, default=2.2)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--outbase", default="results")
    args = ap.parse_args()

    tag = f"enc{args.enc_scale}_L{args.layers}_cut{args.cutoff}_ep{args.epochs}"
    for ds in args.datasets:
        X, y = load_dataset(ds, n=args.n, seed=0)
        for arch in args.archs:
            for s in range(args.seeds):
                seed_dir = Path(args.outbase) / ds / tag / arch / f"seed{s}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                out = seed_dir / "final_summary.json"
                if out.exists():
                    print(f"skip (exists): {out}")
                    continue
                t0 = time.time()
                rec, weights = train_seed(X, y, arch, s, args.epochs, args.layers, args.cutoff, args.enc_scale, args.lr)
                rec["config"] = {
                    "dataset": ds, "arch": arch, "seed": s,
                    "enc_scale": args.enc_scale, "layers": args.layers,
                    "cutoff": args.cutoff, "epochs": args.epochs, "n": args.n,
                    "lr": args.lr, "readout": "mean_photon", "n_modes": 2,
                    "seconds": round(time.time() - t0, 1),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                with open(out, "w") as f:
                    json.dump(rec, f, indent=2)
                np.savez(seed_dir / "weights.npz", **weights)   # for decision-boundary plots
                flag = "  [STILL RISING]" if rec["still_rising"] else ""
                lk = rec["leakage"]["edge_prob_max"]
                lkflag = "  [LEAKAGE]" if lk > 0.01 else ""
                print(f"{ds} {arch} seed{s}: AUC {rec['final_auc']:.3f}  "
                      f"nbar {rec['nbar_entering_first_mid']:.2f}  edgeP {lk:.4f}"
                      f"{flag}{lkflag}  ({rec['config']['seconds']}s)")

    print("\nDone. Records under:", args.outbase)


if __name__ == "__main__":
    main()
