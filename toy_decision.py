#!/usr/bin/env python3
"""
Reads the weights.npz files written by toy_run.py and redraws the Gaussian-vs-Kerr
decision boundary for a chosen config. 

    python toy_decision.py results/rings --enc_scale 2.2 --layers 2 --cutoff 20 --seed 3
"""

import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
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

def run_logits(X, w, arch, layers, cutoff, enc_scale):
    B = X.shape[0]
    eng = sf.Engine("tf", backend_options={"cutoff_dim": cutoff, "batch_size": B})
    prog = build_program(layers, arch)
    args = {"x0": tf.constant(enc_scale*X[:,0], tf.float32),
            "x1": tf.constant(enc_scale*X[:,1], tf.float32)}
    for L in range(layers):
        for k in (f"t1_{L}", f"p1_{L}", f"t2_{L}", f"p2_{L}", f"m0_{L}", f"m1_{L}"):
            args[k] = tf.constant(w[k], tf.float32)
    st = eng.run(prog, args=args).state
    n0 = tf.math.real(st.mean_photon(0)[0]); n1 = tf.math.real(st.mean_photon(1)[0])
    feats = tf.stack([n0, n1], axis=1)
    return (tf.linalg.matvec(feats, tf.constant(w["w"], tf.float32)) + tf.constant(w["b"], tf.float32)).numpy()

def decision_grid(w, arch, layers, cutoff, enc_scale, lim=1.15, res=55):
    xs = np.linspace(-lim, lim, res); ys = np.linspace(-lim, lim, res)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    out = []
    for i in range(0, len(grid), 400):
        out.append(run_logits(grid[i:i+400], w, arch, layers, cutoff, enc_scale))
    Z = 1/(1+np.exp(-np.concatenate(out)))
    return gx, gy, Z.reshape(gx.shape)


def find_config_dir(root, enc_scale, layers, cutoff, epochs):
    """Locate the <tag> dir matching the requested config."""
    for p in root.glob("*"):
        if not p.is_dir():
            continue
        # tag looks like enc2.2_L2_cut20_ep150
        want = f"enc{enc_scale}_L{layers}_cut{cutoff}"
        if p.name.startswith(want) and (epochs is None or f"_ep{epochs}" in p.name):
            return p
    return None


def pick_seed(arch_dir, requested):
    """Return (seed_dir, seed_num, auc). If requested is None, pick best AUC."""
    best = (None, None, -1)
    for sd in sorted(arch_dir.glob("seed*")):
        wf = sd / "weights.npz"
        js = sd / "final_summary.json"
        if not (wf.exists() and js.exists()):
            continue
        snum = int(sd.name.replace("seed", ""))
        if requested is not None and snum != requested:
            continue
        auc = json.loads(js.read_text()).get("final_auc", -1)
        if requested is not None:
            return sd, snum, auc
        if auc > best[2]:
            best = (sd, snum, auc)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="dataset results dir, e.g. results/rings")
    ap.add_argument("--enc_scale", type=float, required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--cutoff", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="specific seed; default = best AUC per arch")
    ap.add_argument("--n", type=int, default=300, help="points to overlay")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    dataset = root.name
    cfg_dir = find_config_dir(root, args.enc_scale, args.layers, args.cutoff, args.epochs)
    if cfg_dir is None:
        raise SystemExit(f"no config dir under {root} matching "
                         f"enc{args.enc_scale}_L{args.layers}_cut{args.cutoff}"
                         + (f"_ep{args.epochs}" if args.epochs else ""))
    print(f"config: {cfg_dir.name}")

    X, y = load_dataset(dataset, n=args.n, seed=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for j, arch in enumerate(["gaussian", "kerr"]):      
        disp = arch.capitalize()                          
        arch_dir = cfg_dir / arch
        if not arch_dir.exists():
            print(f"  (no {disp} runs)"); continue
        sd, snum, auc = pick_seed(arch_dir, args.seed)     
        if sd is None:
            print(f"  (no usable seed for {disp})"); continue
        w = dict(np.load(sd / "weights.npz"))
        gx, gy, Z = decision_grid(w, arch, args.layers, args.cutoff, args.enc_scale)  
        ax = axes[j]
        #ax.contourf(gx, gy, Z, levels=20, cmap="RdBu_r", vmin=0, vmax=1, alpha=0.75)
        levels = np.linspace(0, 1, 31)   # 20 intervals spanning exactly 0 to 1
        cf = ax.contourf(gx, gy, Z, levels=levels, cmap="RdBu_r", vmin=0, vmax=1, alpha=0.75)
        ax.scatter(X[y==0,0], X[y==0,1], s=12, c="navy", linewidths=0.3) 
        ax.scatter(X[y==1,0], X[y==1,1], s=12, c="darkred", linewidths=0.3) 
        ax.set_title(f"{disp} circuit AUC {auc:.3f}", fontsize=15)  
        ax.set_xticks([]); ax.set_yticks([])
        print(f"  {disp}: seed {snum}, AUC {auc:.3f}")

    cbar = fig.colorbar(cf, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_ticks(np.arange(0, 1.01, 0.2))          
    cbar.set_label("Probability", fontsize=13)        
    cbar.ax.tick_params(labelsize=12)

    fig.suptitle(f"{args.layers} layers", fontsize=17)
    #fig.suptitle(f"Encoding scale {args.enc_scale}", fontsize=17)
    fig.tight_layout()
    out = args.out or (cfg_dir / "decision_boundary.pdf")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
