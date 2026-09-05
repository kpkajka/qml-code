#!/usr/bin/env python3
"""
Fit sklearn logistic regression on polynomial expansions of the toy rings features
and report the AUC at each polynomial degree.

python poly_baseline_rings.py --n 6000 --degrees 1 2 3 4 5 6
"""

import argparse
import json
import time
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def make_rings(n, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.uniform(0, 3, n)
    th = rng.uniform(0, 2 * np.pi, n)
    X = np.column_stack([r * np.cos(th), r * np.sin(th)]) / 3.0
    y = (np.floor(r) % 2).astype(int)
    return X, y


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
    ap.add_argument("--n", type=int, default=6000, help="total number of ring points")
    ap.add_argument("--degrees", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="poly_baseline_rings.json")
    args = ap.parse_args()

    X, y = make_rings(args.n, seed=args.seed)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=args.seed, stratify=y)

    print(f"rings  n={args.n}  (class balance: {int(y.sum())}/{len(y)-int(y.sum())})")
    print(f"{'degree':>8}{'AUC':>10}")
    aucs = {}
    for deg in args.degrees:
        a = poly_auc(Xtr, ytr, Xte, yte, deg)
        aucs[str(deg)] = float(a)
        print(f"{deg:>8}{a:>10.4f}")

    result = {
        "dataset": "rings",
        "n_total": int(args.n),
        "seed": args.seed,
        "auc_by_degree": aucs,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
