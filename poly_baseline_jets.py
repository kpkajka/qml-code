#!/usr/bin/env python3
"""
Fit sklearn logistic regression on polynomial expansions of the jet features
(the same top-k eta/phi/pt_norm the quantum runner uses) and report the AUC at
each polynomial degree.

    python poly_baseline_jets.py --path TTBar+ZJets_flat.h5 --modes 6 --n_per_class 3000
"""

import argparse
import json
import time
import numpy as np
import h5py
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def load_data(path):
    with h5py.File(path, "r") as f:
        const = f["jetConstituentsList"][...].astype(np.float32)
        truth = np.rint(f["truth_labels"][...]).astype(np.int64)
        names = [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
                 for x in f["jetFeatureNames"][()]]
        jet_features = f["jetFeatures"][...].astype(np.float32)
    jet_pt = jet_features[:, names.index("jet_pt")].astype(np.float32)
    return const, truth, jet_pt


def build_topk_features(const, jet_pt, indices, k):
    """Identical to the runner's build_topk_features_for_indices, flattened."""
    X = np.zeros((len(indices), k, 3), dtype=np.float32)
    for row, idx in enumerate(indices):
        eta = const[idx, :, 0]; phi = const[idx, :, 1]; pt = const[idx, :, 2]
        valid = np.isfinite(pt) & (pt > 0) & np.isfinite(eta) & np.isfinite(phi)
        eta, phi, pt = eta[valid], phi[valid], pt[valid]
        if pt.size == 0:
            continue
        order = np.argsort(pt)[::-1][:k]
        eta, phi, pt = eta[order], phi[order], pt[order]
        pt_norm = (pt / (jet_pt[idx] + 1e-12)).astype(np.float32)
        n_fill = min(k, pt.size)
        X[row, :n_fill, 0] = eta[:n_fill]
        X[row, :n_fill, 1] = phi[:n_fill]
        X[row, :n_fill, 2] = pt_norm[:n_fill]
    return X.reshape(len(indices), -1)


def balanced_indices(truth, n_per_class, seed):
    rng = np.random.default_rng(seed)
    labels = np.unique(truth)
    bkg, sig = int(labels.min()), int(labels.max())
    bi = np.where(truth == bkg)[0]; si = np.where(truth == sig)[0]
    rng.shuffle(bi); rng.shuffle(si)
    n = min(n_per_class, len(bi), len(si))
    idx = np.concatenate([bi[:n], si[:n]])
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
    p = rng.permutation(len(idx))
    return idx[p], y[p]


def poly_auc(Xtr, ytr, Xte, yte, degree):
    model = make_pipeline(
        PolynomialFeatures(degree, include_bias=False),
        StandardScaler(),
        LogisticRegression(max_iter=5000, C=1.0),
    )
    model.fit(Xtr, ytr)
    return roc_auc_score(yte, model.predict_proba(Xte)[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--modes", type=int, default=6)
    ap.add_argument("--n_per_class", type=int, default=3000)
    ap.add_argument("--degrees", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="poly_baseline_jets_new.json")
    args = ap.parse_args()

    const, truth, jet_pt = load_data(args.path)
    idx, y = balanced_indices(truth, args.n_per_class, args.seed)
    X = build_topk_features(const, jet_pt, idx, k=args.modes)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=args.seed, stratify=y)

    print(f"jets  k={args.modes}  n={len(idx)}")
    print(f"{'degree':>8}{'AUC':>10}")
    aucs = {}
    for deg in args.degrees:
        a = poly_auc(Xtr, ytr, Xte, yte, deg)
        aucs[str(deg)] = float(a)
        print(f"{deg:>8}{a:>10.4f}")

    result = {
        "dataset": "jets",
        "path": args.path,
        "modes": args.modes,
        "n_per_class": args.n_per_class,
        "n_total": int(len(idx)),
        "seed": args.seed,
        "auc_by_degree": aucs,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
