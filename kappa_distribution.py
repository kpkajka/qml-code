#!/usr/bin/env python3
"""
Distribution of the trained Kerr strength (kappa) across seeds,
for the Kerr architectures. Reads architecture__mid_kappa (self-Kerr) and/or
architecture__ck_kappa (cross-Kerr) from each seed's trained_model_weights.npz.

One histogram per architecture (png + pdf). Aggregates all kappa values across
all seeds, layers, and modes/pairs for that architecture.

    python kappa_dist.py classifier/runs --outdir kappa_plots
"""

import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (display name, relpath, which kappa key(s) to pull)
DEFAULT_ARCHES = [
    ("Int-K-Int",    "int_k_int/m6/L1/new_entangled_k/lg_none/cutoff_6",
     ["architecture__mid_kappa"]),
    ("Int-K0-Int",   "int_k0_int/m6/L1/new_entangled_k0/lg_none/cutoff_6",
     ["architecture__mid_kappa0"]),
    ("Int-CK-Int",   "int_kck_int/m6/L1/new_entangled_kck/lg_none/cutoff_6/ck_chain_ckonly",
     ["architecture__ck_kappa"]),
    ("Int-K-CK-Int", "int_kck_int/m6/L1/new_entangled_kck/lg_none/cutoff_6/ck_chain_selfkerr",
     ["architecture__mid_kappa", "architecture__ck_kappa"]),
]
COLORS = {"Int-K-Int": "#C44E52", "Int-K0-Int": "#8172B3",
          "Int-CK-Int": "#55A868", "Int-K-CK-Int": "#CCB974"}


def slug(name):
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def collect_kappa(arch_dir, keys):
    """Flatten all kappa values across seeds/layers/modes for the given keys."""
    vals = []
    for wf in glob.glob(f"{arch_dir}/**/seed*/trained_model_weights.npz", recursive=True):
        try:
            payload = dict(np.load(wf, allow_pickle=True))
        except Exception:
            continue
        for k in keys:
            if k in payload:
                vals.append(np.asarray(payload[k], dtype=np.float64).ravel())
    return np.concatenate(vals) if vals else np.array([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--outdir", default="kappa_plots")
    ap.add_argument("--bins", type=int, default=40)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"{'architecture':<16}{'n_vals':>8}{'median|k|':>11}{'mean k':>9}{'std k':>9}")
    for name, rel, keys in DEFAULT_ARCHES:
        arch_dir = os.path.join(args.root, rel)
        k = collect_kappa(arch_dir, keys)
        if k.size == 0:
            print(f"{name:<16}{'(none found)':>28}")
            continue

if __name__ == "__main__":
    main()
