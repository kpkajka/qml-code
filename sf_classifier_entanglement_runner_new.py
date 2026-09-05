#!/usr/bin/env python3
"""
sf_classifier_entanglement_runner.py

Supervised Strawberry Fields classifier for jet background vs signal studies.

This runner trains a single configuration:
- one mode count
- one entanglement architecture
- one local-gate choice
- one layer count

It is designed to be launched many times from a PBS sweep script.
"""
# Environment setup and imports
import os

# launching everything to single-thread execution and disabling caches (to run on HPC cluster)

os.environ["SYMPY_USE_CACHE"] = "no"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1") 
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import argparse
import json
import time
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import strawberryfields as sf
import sympy.core.cache as sympy_cache
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from strawberryfields import ops
from tqdm.auto import tqdm
import shutil, sys

Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

matplotlib.use("Agg")
sympy_cache.USE_CACHE = False
try:
    sympy_cache.clear_cache()
except Exception:
    pass

tf.get_logger().setLevel("ERROR")


def configure_tf_threading():
    try:
        intra = int(os.environ.get("TF_NUM_INTRAOP_THREADS", os.environ.get("OMP_NUM_THREADS", "1")))
    except Exception:
        intra = 1
    try:
        inter = int(os.environ.get("TF_NUM_INTEROP_THREADS", "1"))
    except Exception:
        inter = 1

    try:
        tf.config.threading.set_intra_op_parallelism_threads(max(intra, 1))
        tf.config.threading.set_inter_op_parallelism_threads(max(inter, 1))
    except Exception:
        pass


configure_tf_threading()


# Global constants and hyperparameters

SEED = 0                   # default seed
CUTOFF = 6                 # number of Fock states that define the Hilbert space
EPOCHS = 100               # cap on number of epochs so it doesnt go on forever
BATCH = 8                  # batch size
LR = 0.01                  # learning rate

# Early stopping settings
PATIENCE = 12 
MIN_DELTA = 1e-4 
LR_PATIENCE = 5           # plateau epochs before dropping LR
LR_FACTOR = 0.5           # multiply LR by this
MIN_LR = 1e-4

# Training, validation, testing split
N_TRAIN_PER_CLASS = 1500
N_VAL_PER_CLASS = 300
N_TEST_PER_CLASS = 500

# Beamsplitter angles (theta=pi/4 gives a 50:50 BS)
BS_THETA = np.pi / 4.0
BS_PHI = np.pi / 2.0

# Cap on mid-circuit squeeze magnitude 
MID_SQ_CAP = 0.55
ENC_DISP_CAP = 2.0

# Entanglement topologies
ARCH_LIST = [
    "fixed_ring",
    "trainable_ring",
    "fixed_alltoall",
    "trainable_alltoall",
    "fixed_chain",
    "trainable_chain",
    "fixed_star",
    "trainable_star",
    "maximally_entangled",
    "new_entangled",
    "new_entangled_k",
    "new_entangled_kck",
    "new_entangled_k0",
    "new_circuit_like",
    "no_entanglement",
]

# Single-mode gate menus
LOCAL_GATES = ["none", "rgate", "sgate", "srgate", "drs", "kerr", "drs_kerr"]

# Measurement schemes
OBSERVABLE_READOUTS = ["mean_photon", "richer_joint"]

STANDARD_BATCHED_ARCH_LIST = [
    "fixed_ring",
    "trainable_ring",
    "fixed_alltoall",
    "trainable_alltoall",
    "fixed_chain",
    "trainable_chain",
    "fixed_star",
    "trainable_star",
    "no_entanglement",
]

# Different ways of turning data into gate parameters
ENCODING_MODES = [
    "trainable_per_mode",
    "trainable_per_mode_kerr",
    "fixed_global_scale",
    "frozen_identity",
    "shared_trainable_linear",
    "shared_trainable_linear_bias",
    "shared_trainable_phase_linear",
    "rgate_only",
]


#Batching and Topology Functions

def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def safe_float(x):
    try:
        return float(x.numpy())
    except Exception:
        return float(x)


# pairs_* defines which pairs of modes get connected by BS

# Each mode links to its neighbour, last one wrapping to the first
def pairs_ring(n):
    return [(i, (i + 1) % n) for i in range(n)]

# Each mode links to its neighbour, no wrapping
def pairs_chain(n):
    return [(i, i + 1) for i in range(n - 1)]

# Each mode to every other mode
def pairs_alltoall(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

# Mode 0 connects to all other modes
def pairs_star(n):
    return [(0, i) for i in range(1, n)]

# Mapping each function to the name
TOPOS = {
    "ring": pairs_ring,
    "chain": pairs_chain,
    "alltoall": pairs_alltoall,
    "star": pairs_star,
}

# Building a mesh pattern
def bs_mesh_pairs_layers(n, depth=None):
    if depth is None:
        depth = n
    layers = []
    for layer_idx in range(depth):
        start = 0 if (layer_idx % 2 == 0) else 1
        pairs = []
        for i in range(start, n - 1, 2):
            pairs.append((i, i + 1))
        layers.append(pairs)
    return layers


def batches(X, y, batch_size, rng):
    idx = np.arange(len(X))
    rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        sl = idx[start : start + batch_size]
        yield X[sl], y[sl]


def sequential_batches(X, y, batch_size):
    for start in range(0, len(X), batch_size):
        stop = start + batch_size
        yield X[start:stop], y[start:stop]


# Data loading and balanced train/validation/test split preparation

def load_data(path):
    with h5py.File(path, "r") as f:
        const = f["jetConstituentsList"][...].astype(np.float32)
        truth = np.rint(f["truth_labels"][...]).astype(np.int64)
        names = [
            x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
            for x in f["jetFeatureNames"][()]
        ]
        jet_features = f["jetFeatures"][...].astype(np.float32)
    jet_pt = jet_features[:, names.index("jet_pt")].astype(np.float32)
    return const, truth, jet_pt


# Sorts jets by k highest pt, output = (n_jets, k, 3)
def build_topk_features_for_indices(const, jet_pt, indices, k):
    X = np.zeros((len(indices), k, 3), dtype=np.float32)
    for row, idx in enumerate(tqdm(indices, desc=f"Building X(k={k})", leave=False)):
        eta = const[idx, :, 0]
        phi = const[idx, :, 1]
        pt = const[idx, :, 2]
        valid = np.isfinite(pt) & (pt > 0) & np.isfinite(eta) & np.isfinite(phi)
        eta, phi, pt = eta[valid], phi[valid], pt[valid]
        if pt.size == 0:
            continue
        order = np.argsort(pt)[::-1][:k]
        eta, phi, pt = eta[order], phi[order], pt[order]
        pt_norm = (pt / (jet_pt[idx] + 1e-12)).astype(np.float32)
        n_fill = min(k, pt.size)
        X[row, :n_fill, 0] = eta[:n_fill].astype(np.float32)
        X[row, :n_fill, 1] = phi[:n_fill].astype(np.float32)
        X[row, :n_fill, 2] = pt_norm[:n_fill]
    return X


# Building class-balanced train/val/test sets from a single file
def prepare_balanced_splits(
    const,
    truth,
    jet_pt,
    n_modes,
    n_train_per_class,
    n_val_per_class,
    n_test_per_class,
    seed,
    train_pool_per_class=None,
):
    labels = np.unique(truth)
    if len(labels) != 2:
        raise RuntimeError(f"Expected binary truth labels, got {labels.tolist()}")

    bkg_label = int(labels.min())
    sig_label = int(labels.max())

    bkg_idx = np.where(truth == bkg_label)[0]
    sig_idx = np.where(truth == sig_label)[0]

    rng = np.random.default_rng(seed)
    rng.shuffle(bkg_idx)
    rng.shuffle(sig_idx)

    requested_pool = n_train_per_class if train_pool_per_class is None else train_pool_per_class
    n_train_pool = min(requested_pool, len(bkg_idx), len(sig_idx))
    n_train = min(n_train_per_class, n_train_pool)
    n_val = min(n_val_per_class, len(bkg_idx) - n_train_pool, len(sig_idx) - n_train_pool)
    n_test = min(
        n_test_per_class,
        len(bkg_idx) - n_train_pool - n_val,
        len(sig_idx) - n_train_pool - n_val,
    )

    if min(n_train, n_val, n_test) <= 0:
        raise RuntimeError("Not enough data to build balanced train/val/test splits")

    bkg_train_pool = bkg_idx[:n_train_pool]
    sig_train_pool = sig_idx[:n_train_pool]
    bkg_train = bkg_train_pool[:n_train]
    sig_train = sig_train_pool[:n_train]
    bkg_val = bkg_idx[n_train_pool : n_train_pool + n_val]
    sig_val = sig_idx[n_train_pool : n_train_pool + n_val]
    bkg_test = bkg_idx[n_train_pool + n_val : n_train_pool + n_val + n_test]
    sig_test = sig_idx[n_train_pool + n_val : n_train_pool + n_val + n_test]

    train_idx = np.concatenate([bkg_train, sig_train])
    val_idx = np.concatenate([bkg_val, sig_val])
    test_idx = np.concatenate([bkg_test, sig_test])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    all_idx = np.unique(np.concatenate([train_idx, val_idx, test_idx]))
    X_all = build_topk_features_for_indices(const, jet_pt, all_idx, k=n_modes)
    idx_to_row = {int(idx): row for row, idx in enumerate(all_idx)}

    def gather(indices):
        rows = np.array([idx_to_row[int(i)] for i in indices], dtype=np.int64)
        return X_all[rows]

    X_train = gather(train_idx)
    X_val = gather(val_idx)
    X_test = gather(test_idx)
    y_train = (truth[train_idx] == sig_label).astype(np.float32)
    y_val = (truth[val_idx] == sig_label).astype(np.float32)
    y_test = (truth[test_idx] == sig_label).astype(np.float32)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "bkg_label": bkg_label,
        "sig_label": sig_label,
        "n_train_pool_per_class": int(n_train_pool),
    }


# Building class-balanced train/val/test sets from separate files
def prepare_balanced_splits_from_files(
    train_const,
    train_truth,
    train_jet_pt,
    val_const,
    val_truth,
    val_jet_pt,
    test_const,
    test_truth,
    test_jet_pt,
    n_modes,
    n_train_per_class,
    n_val_per_class,
    n_test_per_class,
    seed,
):
    train_labels = np.unique(train_truth)
    val_labels = np.unique(val_truth)
    test_labels = np.unique(test_truth)
    if len(train_labels) != 2 or len(val_labels) != 2 or len(test_labels) != 2:
        raise RuntimeError(
            "Expected binary truth labels in train/val/test files "
            f"but got train={train_labels.tolist()} val={val_labels.tolist()} test={test_labels.tolist()}"
        )

    bkg_label = int(min(train_labels.min(), val_labels.min(), test_labels.min()))
    sig_label = int(max(train_labels.max(), val_labels.max(), test_labels.max()))

    rng = np.random.default_rng(seed)

    def pick_indices(truth, n_per_class):
        bkg_idx = np.where(truth == bkg_label)[0]
        sig_idx = np.where(truth == sig_label)[0]
        rng.shuffle(bkg_idx)
        rng.shuffle(sig_idx)
        n_take = min(n_per_class, len(bkg_idx), len(sig_idx))
        if n_take <= 0:
            raise RuntimeError("Not enough data to build balanced splits from provided files")
        return np.concatenate([bkg_idx[:n_take], sig_idx[:n_take]])

    train_idx = pick_indices(train_truth, n_train_per_class)
    val_idx = pick_indices(val_truth, n_val_per_class)
    test_idx = pick_indices(test_truth, n_test_per_class)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    X_train = build_topk_features_for_indices(train_const, train_jet_pt, train_idx, k=n_modes)
    X_val = build_topk_features_for_indices(val_const, val_jet_pt, val_idx, k=n_modes)
    X_test = build_topk_features_for_indices(test_const, test_jet_pt, test_idx, k=n_modes)
    y_train = (train_truth[train_idx] == sig_label).astype(np.float32)
    y_val = (val_truth[val_idx] == sig_label).astype(np.float32)
    y_test = (test_truth[test_idx] == sig_label).astype(np.float32)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "bkg_label": bkg_label,
        "sig_label": sig_label,
        "n_train_pool_per_class": int(min(np.sum(train_truth == bkg_label), np.sum(train_truth == sig_label))),
    }


# Encoder parameter initialisation and input feature encoding

# Squashing
def pos_scale(r):
    return 2.0 * tf.math.sigmoid(r) + 0.05

# Squashing
def pos_amp(r):
    return 5.0 * tf.math.sigmoid(r) + 0.05

def pos_amp_capped(r):
    return ENC_DISP_CAP * tf.math.sigmoid(r) + 0.05   # was 5.0

# Encoder squeezing
def pos_scale_sq(r):
    return 0.5 * tf.math.sigmoid(r) + 0.05     # squeeze magnitude capped at ~0.55

# Various encoding modes
def init_encoder_vars(n_modes, seed, encoding_mode):
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.03, seed=seed)
    if encoding_mode == "trainable_per_mode": 
        return {
            "eta_s": tf.Variable(init((n_modes,)), dtype=tf.float32),
            "eta_b": tf.Variable(tf.zeros((n_modes,)), dtype=tf.float32),
            "phi_s": tf.Variable(init((n_modes,)), dtype=tf.float32),
            "phi_b": tf.Variable(tf.zeros((n_modes,)), dtype=tf.float32),
            "pt_s": tf.Variable(init((n_modes,)), dtype=tf.float32),
            "pt_b": tf.Variable(tf.zeros((n_modes,)), dtype=tf.float32),
            "amp": tf.Variable(np.float32(0.0)),
        }

    if encoding_mode == "trainable_per_mode_kerr":
        return {
            "eta_s": tf.Variable(init((n_modes,)), dtype=tf.float32),
            "eta_b": tf.Variable(tf.zeros((n_modes,)), dtype=tf.float32),
            "phi_s": tf.Variable(init((n_modes,)), dtype=tf.float32),
            "phi_b": tf.Variable(tf.zeros((n_modes,)), dtype=tf.float32),
            "pt_s": tf.Variable(init((n_modes,)), dtype=tf.float32),
            "pt_b": tf.Variable(tf.zeros((n_modes,)), dtype=tf.float32),
            "amp": tf.Variable(np.float32(0.0)),
        }
    
    if encoding_mode == "fixed_global_scale":
        return {
            "global_scale": tf.Variable(np.float32(0.0)),
        }
    if encoding_mode == "frozen_identity":
        return {}
    if encoding_mode == "shared_trainable_linear":
        return {
            "c_eta": tf.Variable(np.float32(1.0)),
            "c_phi": tf.Variable(np.float32(1.0)),
            "c_pt": tf.Variable(np.float32(1.0)),
        }
    if encoding_mode == "shared_trainable_linear_bias":
        return {
            "c_eta": tf.Variable(np.float32(1.0)),
            "c_phi": tf.Variable(np.float32(1.0)),
            "c_pt": tf.Variable(np.float32(1.0)),
            "b_s": tf.Variable(np.float32(0.0)),
            "b_r": tf.Variable(np.float32(0.0)),
            "b_d": tf.Variable(np.float32(0.0)),
        }
    if encoding_mode == "shared_trainable_phase_linear":
        return {
            "c_pt": tf.Variable(np.float32(1.0)),
            "c_d_phi": tf.Variable(np.float32(1.0)),
            "c_eta": tf.Variable(np.float32(1.0)),
            "c_s_phi": tf.Variable(np.float32(1.0)),
            "c_r_phi": tf.Variable(np.float32(1.0)),
        }
    if encoding_mode == "rgate_only":
        return {
            "r_fixed": tf.Variable(np.float32(1.0)),
            "c_eta": tf.Variable(np.float32(1.0)),
            "c_phi": tf.Variable(np.float32(1.0)),
            "c_pt": tf.Variable(np.float32(1.0)),
        }
    raise ValueError(f"Unsupported encoding mode: {encoding_mode}")


# Quantum architecture and local gate construction

# Allocating the trainable circuit parameters for the chosen architecture
def init_arch_vars(arch, pairs, n_modes, n_layers, seed, ck_topology="chain", use_self_kerr=True):
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.03, seed=seed)

    if arch in {
        "fixed_ring", "trainable_ring",
        "fixed_alltoall", "trainable_alltoall",
        "fixed_chain", "trainable_chain",
        "fixed_star", "trainable_star",
    }:
        n_pairs = len(pairs)
        return {
            "theta": tf.Variable(tf.fill((n_layers, n_pairs), tf.constant(BS_THETA, dtype=tf.float32))),
            "phi": tf.Variable(tf.fill((n_layers, n_pairs), tf.constant(BS_PHI, dtype=tf.float32))),
        }

    if arch == "new_entangled":
        mesh_layers = bs_mesh_pairs_layers(n_modes, depth=n_modes)
        n_pairs_total = sum(len(layer_pairs) for layer_pairs in mesh_layers)
        return {
            "mesh_layers": mesh_layers,
            "int1_theta": tf.Variable(init((n_layers, n_pairs_total)), dtype=tf.float32),
            "int1_phi": tf.Variable(init((n_layers, n_pairs_total)), dtype=tf.float32),
            "int2_theta": tf.Variable(init((n_layers, n_pairs_total)), dtype=tf.float32),
            "int2_phi": tf.Variable(init((n_layers, n_pairs_total)), dtype=tf.float32),
            "mid_sq_r": tf.Variable(init((n_layers, n_modes)), dtype=tf.float32),
            "mid_sq_phi": tf.Variable(init((n_layers, n_modes)), dtype=tf.float32),
        }

    
    if arch == "new_entangled_kck":
        # Int-K-CK-Int: self-Kerr + cross-Kerr between the interferometers
        if not hasattr(ops, "CKgate"):
            raise SystemExit("This Strawberry Fields build has no ops.CKgate (cross-Kerr).")
        mesh_layers = bs_mesh_pairs_layers(n_modes, depth=n_modes)
        n_pairs_total = sum(len(layer_pairs) for layer_pairs in mesh_layers)
        ck_pairs = TOPOS[ck_topology](n_modes)
        out = {
            "mesh_layers": mesh_layers,
            "ck_pairs": ck_pairs,
            "int1_theta": tf.Variable(init((n_layers, n_pairs_total)), dtype=tf.float32),
            "int1_phi":   tf.Variable(init((n_layers, n_pairs_total)), dtype=tf.float32),
            "int2_theta": tf.Variable(init((n_layers, n_pairs_total)), dtype=tf.float32),
            "int2_phi":   tf.Variable(init((n_layers, n_pairs_total)), dtype=tf.float32),
            "ck_kappa":   tf.Variable(init((n_layers, len(ck_pairs))), dtype=tf.float32),
        }
        if use_self_kerr:
            out["mid_kappa"] = tf.Variable(init((n_layers, n_modes)), dtype=tf.float32)
        return out


    if arch == "new_entangled_k":
        # Int-K-Int: single-mode Kerr in place of the mid squeeze
        mesh_layers = bs_mesh_pairs_layers(n_modes, depth=n_modes)
        n_pairs_total = sum(len(layer_pairs) for layer_pairs in mesh_layers)
        return {
            "mesh_layers": mesh_layers,
            "int1_theta": tf.Variable(init((n_layers, n_pairs_total)), dtype=tf.float32),
            "int1_phi": tf.Variable(init((n_layers, n_pairs_total)), dtype=tf.float32),
            "int2_theta": tf.Variable(init((n_layers, n_pairs_total)), dtype=tf.float32),
            "int2_phi": tf.Variable(init((n_layers, n_pairs_total)), dtype=tf.float32),
            # single-mode Kerr: K(k) = exp(i * k * n^2)
            "mid_kappa": tf.Variable(init((n_layers, n_modes)), dtype=tf.float32),
        }
 

    if arch == "new_entangled_k0":
        # Single mode kerr on 0-th mode, squeeze on rest
        mesh_layers = bs_mesh_pairs_layers(n_modes, depth=n_modes)
        n_pairs_total = sum(len(layer_pairs) for layer_pairs in mesh_layers)
        return {
            "mesh_layers": mesh_layers,
            "int1_theta": tf.Variable(init((n_layers, n_pairs_total)), dtype=tf.float32),
            "int1_phi":   tf.Variable(init((n_layers, n_pairs_total)), dtype=tf.float32),
            "int2_theta": tf.Variable(init((n_layers, n_pairs_total)), dtype=tf.float32),
            "int2_phi":   tf.Variable(init((n_layers, n_pairs_total)), dtype=tf.float32),
            # squeeze on modes 1..n-1 only; mode 0 gets a Kerr instead
            "mid_sq_r":   tf.Variable(init((n_layers, n_modes)), dtype=tf.float32),
            "mid_sq_phi": tf.Variable(init((n_layers, n_modes)), dtype=tf.float32),
            "mid_kappa0": tf.Variable(init((n_layers, 1)), dtype=tf.float32),
        }
    

    if arch == "new_circuit_like":
        all_pairs = pairs_alltoall(n_modes)
        return {
            "all_pairs": all_pairs,
            "bs_all_theta": tf.Variable(init((n_layers, len(all_pairs))), dtype=tf.float32),
            "bs_all_phi": tf.Variable(init((n_layers, len(all_pairs))), dtype=tf.float32),
        }

    return {}

# Creating the single-mode trainable gates chosen by --local_gates
def init_local_vars(n_modes, n_layers, local_gates, seed):
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.03, seed=seed)
    vars_out = {}
    if local_gates in ("rgate", "srgate", "drs", "drs_kerr"):
        vars_out["r_ang"] = tf.Variable(init((n_layers, n_modes)), dtype=tf.float32)
    if local_gates in ("sgate", "srgate", "drs", "drs_kerr"):
        vars_out["s_r"] = tf.Variable(init((n_layers, n_modes)), dtype=tf.float32)
    if local_gates in ("drs", "drs_kerr"):
        vars_out["disp_r"] = tf.Variable(init((n_layers, n_modes)), dtype=tf.float32)
        vars_out["disp_phi"] = tf.Variable(init((n_layers, n_modes)), dtype=tf.float32)
    if local_gates in ("kerr", "drs_kerr"):
        vars_out["kappa"] = tf.Variable(init((n_layers, n_modes)), dtype=tf.float32)
    return vars_out

# Making the final linear layer's weights w and bias b, sized to match the readout
def init_classifier_head(n_modes, seed):
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed)
    return {
        "w": tf.Variable(init((n_modes,)), dtype=tf.float32),
        "b": tf.Variable(np.float32(0.0)),
    }

# Computing the size of the readout
def observable_feature_dim(n_modes, observable_readout):
    if observable_readout == "mean_photon":
        return n_modes
    if observable_readout == "richer_joint":
        return n_modes + n_modes + (n_modes * (n_modes - 1) // 2)
    raise ValueError(f"Unsupported observable readout: {observable_readout}")

# Making the final linear layer's weights w and bias b, sized to match the readout
def init_classifier_head_for_readout(n_modes, seed, observable_readout):
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed)
    n_features = observable_feature_dim(n_modes, observable_readout)
    return {
        "w": tf.Variable(init((n_features,)), dtype=tf.float32),
        "b": tf.Variable(np.float32(0.0)),
    }


def encoding_mode_description(encoding_mode):
    if encoding_mode == "trainable_per_mode":
        return (
            "Per-mode trainable affine encoder for eta, phi, and pt_norm, "
            "plus a learned global displacement scale."
        )
    if encoding_mode == "fixed_global_scale":
        return (
            "Fixed shared encoder across all modes: eta and phi use identity mapping, "
            "pt_norm is clipped to [0, 5], and only one learned global displacement scale is trainable."
        )
    if encoding_mode == "frozen_identity":
        return (
            "Fully frozen shared encoder across all modes: eta and phi use identity mapping, "
            "pt_norm is clipped to [0, 5], and the displacement amplitude uses the same fixed identity mapping "
            "with no trainable encoder parameters."
        )
    if encoding_mode == "shared_trainable_linear":
        return (
            "Shared trainable encoder across all modes: D(c_pt * pt_norm, 0), "
            "S(c_eta * eta, 0), and R(c_phi * phi), with one learned global coefficient per feature branch "
            "and no per-mode encoder parameters."
        )
    if encoding_mode == "shared_trainable_linear_bias":
        return (
            "Shared trainable encoder across all modes with one learned scale and one learned bias per gate branch: "
            "D(c_pt * pt_norm + b_d, 0), S(c_eta * eta + b_s, 0), and R(c_phi * phi + b_r)."
        )
    if encoding_mode == "shared_trainable_phase_linear":
        return (
            "Shared trainable phase-aware encoder across all modes with no learned bias terms: "
            "D(c_pt * pt_norm, c_d_phi * phi), S(c_eta * eta, c_s_phi * phi), "
            "and R(c_r_phi * phi)."
        )
    if encoding_mode == "rgate_only":
        return (
            "Shared trainable encoder across all modes: first apply the same trainable squeezing "
            "S(r_fixed, 0) to every mode, then apply one data-dependent rotation "
            "R(c_pt * clip(pt_norm, 0, 5) + c_eta * eta + c_phi * phi) per mode, with no displacement gate."
        )
    if encoding_mode == "trainable_per_mode_kerr":
        return (
            "Per-mode trainable affine encoder for eta and pt_norm, with phi "
            "encoded via a Kerr gate K(c_phi * phi + b_phi) in place of the rotation, "
            "plus a learned global displacement scale."
        )
    raise ValueError(f"Unsupported encoding mode: {encoding_mode}")


def compute_encoder_gate_args(x, enc_vars, encoding_mode):
    x = tf.convert_to_tensor(x, dtype=tf.float32)

    if encoding_mode == "trainable_per_mode":
        eta_scale = pos_scale_sq(enc_vars["eta_s"])     
        #eta_scale = pos_scale(enc_vars["eta_s"])
        phi_scale = pos_scale(enc_vars["phi_s"])
        pt_scale = pos_scale(enc_vars["pt_s"])
        eta = eta_scale * x[..., :, 0] + enc_vars["eta_b"]
        phi = phi_scale * x[..., :, 1] + enc_vars["phi_b"]
        pt = tf.clip_by_value(pt_scale * x[..., :, 2] + enc_vars["pt_b"], 0.0, 5.0)
        disp = pos_amp(enc_vars["amp"]) * pt
        return disp, tf.zeros_like(disp), eta, tf.zeros_like(eta), phi

    if encoding_mode == "trainable_per_mode_kerr":
        eta_scale = pos_scale_sq(enc_vars["eta_s"])
        phi_scale = pos_scale_kerr(enc_vars["phi_s"])
        pt_scale = pos_scale(enc_vars["pt_s"])
        eta = eta_scale * x[..., :, 0] + enc_vars["eta_b"]
        kappa = phi_scale * x[..., :, 1] + enc_vars["phi_b"]     # Kerr strength encodes phi
        pt = tf.clip_by_value(pt_scale * x[..., :, 2] + enc_vars["pt_b"], 0.0, 5.0)
        disp = pos_amp_capped(enc_vars["amp"]) * pt
        return disp, tf.zeros_like(disp), eta, tf.zeros_like(eta), kappa

    if encoding_mode == "fixed_global_scale":
        pt = tf.clip_by_value(x[..., :, 2], 0.0, 5.0)
        disp = pos_amp(enc_vars["global_scale"]) * pt
        eta = x[..., :, 0]
        phi = x[..., :, 1]
        return disp, tf.zeros_like(disp), eta, tf.zeros_like(eta), phi

    if encoding_mode == "frozen_identity":
        pt = tf.clip_by_value(x[..., :, 2], 0.0, 5.0)
        eta = x[..., :, 0]
        phi = x[..., :, 1]
        return pt, tf.zeros_like(pt), eta, tf.zeros_like(eta), phi

    if encoding_mode == "shared_trainable_linear":
        pt = tf.clip_by_value(x[..., :, 2], 0.0, 5.0)
        disp = enc_vars["c_pt"] * pt
        eta = enc_vars["c_eta"] * x[..., :, 0]
        phi = enc_vars["c_phi"] * x[..., :, 1]
        return disp, tf.zeros_like(disp), eta, tf.zeros_like(eta), phi

    if encoding_mode == "shared_trainable_linear_bias":
        pt = tf.clip_by_value(x[..., :, 2], 0.0, 5.0)
        disp = enc_vars["c_pt"] * pt + enc_vars["b_d"]
        eta = enc_vars["c_eta"] * x[..., :, 0] + enc_vars["b_s"]
        phi = enc_vars["c_phi"] * x[..., :, 1] + enc_vars["b_r"]
        return disp, tf.zeros_like(disp), eta, tf.zeros_like(eta), phi

    if encoding_mode == "shared_trainable_phase_linear":
        pt = tf.clip_by_value(x[..., :, 2], 0.0, 5.0)
        phi_feature = x[..., :, 1]
        disp = enc_vars["c_pt"] * pt
        disp_phi = enc_vars["c_d_phi"] * phi_feature
        sq = enc_vars["c_eta"] * x[..., :, 0]
        sq_phi = enc_vars["c_s_phi"] * phi_feature
        rot = enc_vars["c_r_phi"] * phi_feature
        return disp, disp_phi, sq, sq_phi, rot

    if encoding_mode == "rgate_only":
        pt = tf.clip_by_value(x[..., :, 2], 0.0, 5.0)
        disp = tf.zeros_like(pt)
        sq = tf.ones_like(pt) * enc_vars["r_fixed"]
        rot = (
            enc_vars["c_pt"] * pt
            + enc_vars["c_eta"] * x[..., :, 0]
            + enc_vars["c_phi"] * x[..., :, 1]
        )
        return disp, tf.zeros_like(disp), sq, tf.zeros_like(sq), rot

    raise ValueError(f"Unsupported encoding mode: {encoding_mode}")


def encode_input(q, x, n_modes, enc_vars, encoding_mode):
    enc_disp, enc_disp_phi, enc_sq, enc_sq_phi, enc_last = compute_encoder_gate_args(x, enc_vars, encoding_mode)

    for w in range(n_modes):
        ops.Dgate(enc_disp[w], enc_disp_phi[w]) | q[w]
        ops.Sgate(enc_sq[w], enc_sq_phi[w]) | q[w]
        #ops.Rgate(enc_rot[w]) | q[w]
        if encoding_mode == "trainable_per_mode_kerr":
            ops.Kgate(enc_last[w]) | q[w]        # D, S, K: Kerr encodes phi
        else:
            ops.Rgate(enc_last[w]) | q[w]        # D, S, R: rotation (all other modes)


def apply_local_gates(q, n_modes, layer_idx, local_vars, local_gates):
    if local_gates == "none" or not local_vars:
        return
    for w in range(n_modes):
        if local_gates in ("drs", "drs_kerr"):
            ops.Dgate(local_vars["disp_r"][layer_idx, w], local_vars["disp_phi"][layer_idx, w]) | q[w]
        if local_gates in ("rgate", "srgate", "drs", "drs_kerr"):
            ops.Rgate(local_vars["r_ang"][layer_idx, w]) | q[w]
        if local_gates in ("sgate", "srgate", "drs", "drs_kerr"):
            ops.Sgate(local_vars["s_r"][layer_idx, w], 0.0) | q[w]
        if local_gates in ("kerr", "drs_kerr"):
            ops.Kgate(local_vars["kappa"][layer_idx, w]) | q[w]


# Defining each architecture

# q[i] is the qumode
# arch is the architecture name
# outer loop = no. of layers
# inner loop = internal depth fixed at n_modes

def apply_architecture(q, arch, pairs, n_modes, n_layers, arch_vars, local_vars, local_gates):
    
    # Only local gates applied, no entanglement between qumodes
    if arch == "no_entanglement":
        for layer_idx in range(n_layers):
            apply_local_gates(q, n_modes, layer_idx, local_vars, local_gates)
        return

    # Fixed mesh of 50:50 BS (no trainable params)
    if arch == "maximally_entangled":
        mesh_layers = bs_mesh_pairs_layers(n_modes, depth=n_modes)
        for layer_idx in range(n_layers): # Block repetition
            for layer_pairs in mesh_layers: # Column repetiion
                for a, b in layer_pairs: # Each pair in the column
                    ops.BSgate(BS_THETA, BS_PHI) | (q[a], q[b]) # fixed BS
                    # | is the apply operator (here mixing two modes)
            apply_local_gates(q, n_modes, layer_idx, local_vars, local_gates)
        return

    # Int-S-Int architecture
    if arch == "new_entangled":
        mesh_layers = arch_vars["mesh_layers"]
        for layer_idx in range(n_layers): # Block repetition
            # First interferometer applies a trainable BS
            pair_idx = 0
            for layer_pairs in mesh_layers:
                for a, b in layer_pairs:
                    ops.BSgate(arch_vars["int1_theta"][layer_idx, pair_idx], arch_vars["int1_phi"][layer_idx, pair_idx]) | (q[a], q[b])
                    pair_idx += 1
            # Trainable single-mode squeezing         
            for w in range(n_modes):
                #ops.Sgate(arch_vars["mid_sq_r"][layer_idx, w], arch_vars["mid_sq_phi"][layer_idx, w]) | q[w]
                ops.Sgate(MID_SQ_CAP * tf.math.tanh(arch_vars["mid_sq_r"][layer_idx, w]), arch_vars["mid_sq_phi"][layer_idx, w]) | q[w]
            # Second interferometer reading from int2
            pair_idx = 0
            for layer_pairs in mesh_layers:
                for a, b in layer_pairs:
                    ops.BSgate(arch_vars["int2_theta"][layer_idx, pair_idx], arch_vars["int2_phi"][layer_idx, pair_idx]) | (q[a], q[b])
                    pair_idx += 1
            apply_local_gates(q, n_modes, layer_idx, local_vars, local_gates)
        return
    

    # Int-K-Int: non-Gaussian core between the two interferometers
    if arch == "new_entangled_kck":
        mesh_layers = arch_vars["mesh_layers"]
        for layer_idx in range(n_layers):
            # Interferometer 1
            pair_idx = 0
            for layer_pairs in mesh_layers:
                for a, b in layer_pairs:
                    ops.BSgate(arch_vars["int1_theta"][layer_idx, pair_idx], arch_vars["int1_phi"][layer_idx, pair_idx]) | (q[a], q[b])
                    pair_idx += 1
 
            if "mid_kappa" in arch_vars:
                for w in range(n_modes):
                    ops.Kgate(arch_vars["mid_kappa"][layer_idx, w]) | q[w]
            for pair_idx, (a, b) in enumerate(arch_vars["ck_pairs"]):
                ops.CKgate(arch_vars["ck_kappa"][layer_idx, pair_idx]) | (q[a], q[b])
 
            # Interferometer 2 -- makes the Kerr phases observable
            pair_idx = 0
            for layer_pairs in mesh_layers:
                for a, b in layer_pairs:
                    ops.BSgate(arch_vars["int2_theta"][layer_idx, pair_idx], arch_vars["int2_phi"][layer_idx, pair_idx]) | (q[a], q[b])
                    pair_idx += 1
 
            apply_local_gates(q, n_modes, layer_idx, local_vars, local_gates)
        return



    # Int-K-Int: single-mode Kerr between the two interferometers (no squeeze)
    if arch == "new_entangled_k":
        mesh_layers = arch_vars["mesh_layers"]
        for layer_idx in range(n_layers):
            # Interferometer 1
            pair_idx = 0
            for layer_pairs in mesh_layers:
                for a, b in layer_pairs:
                    ops.BSgate(arch_vars["int1_theta"][layer_idx, pair_idx], arch_vars["int1_phi"][layer_idx, pair_idx]) | (q[a], q[b])
                    pair_idx += 1
 
            # Single-mode Kerr in place of the squeeze (phase-only, non-Gaussian)
            for w in range(n_modes):
                ops.Kgate(arch_vars["mid_kappa"][layer_idx, w]) | q[w]
 
            # Interferometer 2 -- makes the Kerr phases observable
            pair_idx = 0
            for layer_pairs in mesh_layers:
                for a, b in layer_pairs:
                    ops.BSgate(arch_vars["int2_theta"][layer_idx, pair_idx], arch_vars["int2_phi"][layer_idx, pair_idx]) | (q[a], q[b])
                    pair_idx += 1
 
            apply_local_gates(q, n_modes, layer_idx, local_vars, local_gates)
        return
 
    # kerr in mode 0, squeeze in modes 1-5
    if arch == "new_entangled_k0":
        mesh_layers = arch_vars["mesh_layers"]
        for layer_idx in range(n_layers):
            # Interferometer 1
            pair_idx = 0
            for layer_pairs in mesh_layers:
                for a, b in layer_pairs:
                    ops.BSgate(arch_vars["int1_theta"][layer_idx, pair_idx],
                               arch_vars["int1_phi"][layer_idx, pair_idx]) | (q[a], q[b])
                    pair_idx += 1

            # --- mixed middle: Kerr on mode 0, squeeze on the rest ---
            ops.Kgate(arch_vars["mid_kappa0"][layer_idx, 0]) | q[0]
            for w in range(1, n_modes):
                ops.Sgate(MID_SQ_CAP * tf.math.tanh(arch_vars["mid_sq_r"][layer_idx, w]),
                          arch_vars["mid_sq_phi"][layer_idx, w]) | q[w]

            # Interferometer 2
            pair_idx = 0
            for layer_pairs in mesh_layers:
                for a, b in layer_pairs:
                    ops.BSgate(arch_vars["int2_theta"][layer_idx, pair_idx],
                               arch_vars["int2_phi"][layer_idx, pair_idx]) | (q[a], q[b])
                    pair_idx += 1

            apply_local_gates(q, n_modes, layer_idx, local_vars, local_gates)
        return

    
    # Ring entanglement with a trainable BS on each pair
    if arch == "new_circuit_like":
        ring_pairs = pairs_ring(n_modes)
        for layer_idx in range(n_layers):
            for pair_idx, (a, b) in enumerate(arch_vars["all_pairs"]):
                ops.BSgate(arch_vars["bs_all_theta"][layer_idx, pair_idx], arch_vars["bs_all_phi"][layer_idx, pair_idx]) | (q[a], q[b])
            for a, b in ring_pairs:
                ops.BSgate(BS_THETA, BS_PHI) | (q[a], q[b])
            apply_local_gates(q, n_modes, layer_idx, local_vars, local_gates)
        return

    # All architectures starting with "trainable"
    trainable = arch.startswith("trainable_")
    for layer_idx in range(n_layers):
        for pair_idx, (a, b) in enumerate(pairs):
            theta = arch_vars["theta"][layer_idx, pair_idx] if trainable else BS_THETA
            phi = arch_vars["phi"][layer_idx, pair_idx] if trainable else BS_PHI
            ops.BSgate(theta, phi) | (q[a], q[b])
        apply_local_gates(q, n_modes, layer_idx, local_vars, local_gates)


def mean_photon_from_probs(probs, mode, cutoff, n_modes):
    photon_numbers = tf.cast(tf.range(cutoff), probs.dtype)
    axes = [ax for ax in range(n_modes) if ax != mode]
    marginal = tf.reduce_sum(probs, axis=axes)
    return tf.reduce_sum(marginal * photon_numbers)


def run_circuit(x, n_modes, arch, pairs, n_layers, cutoff, enc_vars, arch_vars, local_vars, local_gates, engine, encoding_mode):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    prog = sf.Program(n_modes)
    with prog.context as q:
        encode_input(q, x, n_modes, enc_vars, encoding_mode)
        apply_architecture(q, arch, pairs, n_modes, n_layers, arch_vars, local_vars, local_gates)
    state = engine.run(prog).state
    probs = state.all_fock_probs()
    engine.reset()
    return probs


# Batched backend to speed up training

def supports_batched_backend(arch):
    return arch in STANDARD_BATCHED_ARCH_LIST


def build_batched_program(n_modes, arch, pairs, n_layers, local_gates):
    prog = sf.Program(n_modes)
    with prog.context as q:
        for w in range(n_modes):
            ops.Dgate(prog.params(f"enc_disp_{w}"), prog.params(f"enc_disp_phi_{w}")) | q[w]
            ops.Sgate(prog.params(f"enc_sq_{w}"), prog.params(f"enc_sq_phi_{w}")) | q[w]
            ops.Rgate(prog.params(f"enc_rot_{w}")) | q[w]

        for layer_idx in range(n_layers):
            if arch != "no_entanglement":
                for pair_idx, (a, b) in enumerate(pairs):
                    ops.BSgate(
                        prog.params(f"bs_theta_{layer_idx}_{pair_idx}"),
                        prog.params(f"bs_phi_{layer_idx}_{pair_idx}"),
                    ) | (q[a], q[b])

            if local_gates in ("rgate", "sgate", "srgate", "drs", "kerr", "drs_kerr"):
                for w in range(n_modes):
                    if local_gates in ("drs", "drs_kerr"):
                        ops.Dgate(
                            prog.params(f"local_d_r_{layer_idx}_{w}"),
                            prog.params(f"local_d_phi_{layer_idx}_{w}"),
                        ) | q[w]
                    if local_gates in ("rgate", "srgate", "drs", "drs_kerr"):
                        ops.Rgate(prog.params(f"local_r_{layer_idx}_{w}")) | q[w]
                    if local_gates in ("sgate", "srgate", "drs", "drs_kerr"):
                        ops.Sgate(prog.params(f"local_s_{layer_idx}_{w}"), 0.0) | q[w]
                    if local_gates in ("kerr", "drs_kerr"):
                        ops.Kgate(prog.params(f"local_kappa_{layer_idx}_{w}")) | q[w]

    return prog


def build_batched_feed(Xb, n_modes, arch, pairs, n_layers, enc_vars, arch_vars, local_vars, local_gates, encoding_mode):
    Xf = tf.convert_to_tensor(Xb, dtype=tf.float32)
    batch_size = int(Xf.shape[0])
    feed = {}

    enc_disp, enc_disp_phi, enc_sq, enc_sq_phi, enc_rot = compute_encoder_gate_args(Xf, enc_vars, encoding_mode)

    for w in range(n_modes):
        feed[f"enc_disp_{w}"] = tf.cast(enc_disp[:, w], tf.float32)
        feed[f"enc_disp_phi_{w}"] = tf.cast(enc_disp_phi[:, w], tf.float32)
        feed[f"enc_sq_{w}"] = tf.cast(enc_sq[:, w], tf.float32)
        feed[f"enc_sq_phi_{w}"] = tf.cast(enc_sq_phi[:, w], tf.float32)
        feed[f"enc_rot_{w}"] = tf.cast(enc_rot[:, w], tf.float32)

    if arch != "no_entanglement":
        trainable = arch.startswith("trainable_")
        fixed_theta = tf.constant(BS_THETA, dtype=tf.float32)
        fixed_phi = tf.constant(BS_PHI, dtype=tf.float32)
        for layer_idx in range(n_layers):
            for pair_idx, _ in enumerate(pairs):
                theta = arch_vars["theta"][layer_idx, pair_idx] if trainable else fixed_theta
                phi = arch_vars["phi"][layer_idx, pair_idx] if trainable else fixed_phi
                feed[f"bs_theta_{layer_idx}_{pair_idx}"] = tf.broadcast_to(
                    tf.cast(theta, tf.float32), [batch_size]
                )
                feed[f"bs_phi_{layer_idx}_{pair_idx}"] = tf.broadcast_to(
                    tf.cast(phi, tf.float32), [batch_size]
                )

    if local_gates in ("rgate", "sgate", "srgate", "drs", "kerr", "drs_kerr"):
        for layer_idx in range(n_layers):
            for w in range(n_modes):
                if local_gates in ("drs", "drs_kerr"):
                    feed[f"local_d_r_{layer_idx}_{w}"] = tf.broadcast_to(
                        tf.cast(local_vars["disp_r"][layer_idx, w], tf.float32), [batch_size]
                    )
                    feed[f"local_d_phi_{layer_idx}_{w}"] = tf.broadcast_to(
                        tf.cast(local_vars["disp_phi"][layer_idx, w], tf.float32), [batch_size]
                    )
                if local_gates in ("rgate", "srgate", "drs", "drs_kerr"):
                    feed[f"local_r_{layer_idx}_{w}"] = tf.broadcast_to(
                        tf.cast(local_vars["r_ang"][layer_idx, w], tf.float32), [batch_size]
                    )
                if local_gates in ("sgate", "srgate", "drs", "drs_kerr"):
                    feed[f"local_s_{layer_idx}_{w}"] = tf.broadcast_to(
                        tf.cast(local_vars["s_r"][layer_idx, w], tf.float32), [batch_size]
                    )
                if local_gates in ("kerr", "drs_kerr"):
                    feed[f"local_kappa_{layer_idx}_{w}"] = tf.broadcast_to(
                        tf.cast(local_vars["kappa"][layer_idx, w], tf.float32), [batch_size]
                    )

    return feed


# Photon readout and classifier logits
# converts the quantum output into a class score
def photon_logits_from_probs(probs, clf_head, cutoff, n_modes):
    means = tf.stack(
        [tf.cast(mean_photon_from_probs(probs, m, cutoff, n_modes), tf.float32) for m in range(n_modes)]
    )
    return tf.tensordot(clf_head["w"], means, axes=1) + clf_head["b"]


def photon_features_from_probs(probs, cutoff, n_modes, observable_readout):
    photon_numbers = tf.cast(tf.range(cutoff), probs.dtype)
    means = []
    variances = []
    total_axes = tuple(range(n_modes))
    for mode in range(n_modes):
        axes = [ax for ax in range(n_modes) if ax != mode]
        marginal = tf.reduce_sum(probs, axis=axes)
        mean = tf.reduce_sum(marginal * photon_numbers)
        second = tf.reduce_sum(marginal * tf.square(photon_numbers))
        means.append(tf.cast(mean, tf.float32))
        variances.append(tf.cast(second - tf.square(mean), tf.float32))

    if observable_readout == "mean_photon":
        return tf.stack(means)

    if observable_readout != "richer_joint":
        raise ValueError(f"Unsupported observable readout: {observable_readout}")

    pair_covariances = []
    for i in range(n_modes):
        for j in range(i + 1, n_modes):
            pair_moment = tf.reduce_sum(
                probs
                * mode_number_weight(photon_numbers, i, n_modes)
                * mode_number_weight(photon_numbers, j, n_modes),
                axis=total_axes,
            )
            pair_covariances.append(tf.cast(pair_moment, tf.float32) - means[i] * means[j])

    return tf.concat([tf.stack(means), tf.stack(variances), tf.stack(pair_covariances)], axis=0)


def photon_logits_from_probs_readout(probs, clf_head, cutoff, n_modes, observable_readout):
    features = photon_features_from_probs(probs, cutoff, n_modes, observable_readout)
    return tf.tensordot(clf_head["w"], features, axes=1) + clf_head["b"]

# Diagnostic for whether the CUTOFF is too low 
def fock_truncation_stats_from_probs(probs, cutoff, n_modes):
    total_probability = tf.reduce_sum(probs)
    edge_probabilities = []
    for mode in range(n_modes):
        slices = [slice(None)] * n_modes
        slices[mode] = cutoff - 1
        edge_probabilities.append(tf.reduce_sum(probs[tuple(slices)]))
    interior_slices = tuple(slice(0, cutoff - 1) for _ in range(n_modes))
    any_edge_probability = total_probability - tf.reduce_sum(probs[interior_slices])
    return {
        "total_probability": total_probability,
        "any_edge_probability": any_edge_probability,
        "per_mode_edge_probability": tf.stack(edge_probabilities),
    }


def mode_number_weight_batch(photon_numbers, mode, n_modes):
    shape = [1] * (n_modes + 1)
    shape[mode + 1] = photon_numbers.shape[0]
    return tf.reshape(photon_numbers, shape)


def mode_number_weight(photon_numbers, mode, n_modes):
    shape = [1] * n_modes
    shape[mode] = photon_numbers.shape[0]
    return tf.reshape(photon_numbers, shape)


def mean_photon_from_probs_batch(probs, mode, cutoff, n_modes):
    photon_numbers = tf.cast(tf.range(cutoff), probs.dtype)
    weight = mode_number_weight_batch(photon_numbers, mode, n_modes)
    total_axes = tuple(range(1, n_modes + 1))
    return tf.reduce_sum(probs * weight, axis=total_axes)


def photon_logits_from_probs_batch(probs, clf_head, cutoff, n_modes):
    means = tf.stack(
        [tf.cast(mean_photon_from_probs_batch(probs, m, cutoff, n_modes), tf.float32) for m in range(n_modes)],
        axis=1,
    )
    return tf.linalg.matvec(means, tf.cast(clf_head["w"], tf.float32)) + clf_head["b"]


def photon_features_from_probs_batch(probs, cutoff, n_modes, observable_readout):
    photon_numbers = tf.cast(tf.range(cutoff), probs.dtype)
    means = []
    variances = []
    for mode in range(n_modes):
        axes = tuple(ax for ax in range(1, n_modes + 1) if ax != mode + 1)
        marginal = tf.reduce_sum(probs, axis=axes)
        mean = tf.reduce_sum(marginal * tf.reshape(photon_numbers, [1, cutoff]), axis=1)
        second = tf.reduce_sum(marginal * tf.reshape(tf.square(photon_numbers), [1, cutoff]), axis=1)
        means.append(tf.cast(mean, tf.float32))
        variances.append(tf.cast(second - tf.square(mean), tf.float32))

    mean_features = tf.stack(means, axis=1)
    if observable_readout == "mean_photon":
        return mean_features

    if observable_readout != "richer_joint":
        raise ValueError(f"Unsupported observable readout: {observable_readout}")

    pair_covariances = []
    total_axes = tuple(range(1, n_modes + 1))
    for i in range(n_modes):
        for j in range(i + 1, n_modes):
            pair_moment = tf.reduce_sum(
                probs
                * mode_number_weight_batch(photon_numbers, i, n_modes)
                * mode_number_weight_batch(photon_numbers, j, n_modes),
                axis=total_axes,
            )
            pair_covariances.append(tf.cast(pair_moment, tf.float32) - means[i] * means[j])

    return tf.concat(
        [mean_features, tf.stack(variances, axis=1), tf.stack(pair_covariances, axis=1)],
        axis=1,
    )


def photon_logits_from_probs_batch_readout(probs, clf_head, cutoff, n_modes, observable_readout):
    features = photon_features_from_probs_batch(probs, cutoff, n_modes, observable_readout)
    return tf.linalg.matvec(features, tf.cast(clf_head["w"], tf.float32)) + clf_head["b"]


def fock_truncation_stats_from_probs_batch(probs, cutoff, n_modes):
    total_axes = tuple(range(1, n_modes + 1))
    total_probability = tf.reduce_sum(probs, axis=total_axes)
    edge_probabilities = []
    for mode in range(n_modes):
        # Use gather instead of Python tensor slicing so high-rank batched tensors
        # stay on a supported code path and reduction axes match the sliced rank.
        edge_slice = tf.gather(probs, cutoff - 1, axis=mode + 1)
        edge_axes = tuple(range(1, n_modes))
        edge_probabilities.append(tf.reduce_sum(edge_slice, axis=edge_axes) if edge_axes else edge_slice)

    interior_probs = probs
    for mode in range(n_modes):
        interior_probs = tf.gather(interior_probs, tf.range(cutoff - 1), axis=mode + 1)
    any_edge_probability = total_probability - tf.reduce_sum(interior_probs, axis=total_axes)
    return {
        "total_probability": total_probability,
        "any_edge_probability": any_edge_probability,
        "per_mode_edge_probability": tf.stack(edge_probabilities, axis=1),
    }


def summarize_truncation_stats(total_probability, any_edge_probability, per_mode_edge_probability):
    total_probability = np.asarray(total_probability, dtype=np.float64)
    any_edge_probability = np.asarray(any_edge_probability, dtype=np.float64)
    per_mode_edge_probability = np.asarray(per_mode_edge_probability, dtype=np.float64)
    if total_probability.size == 0:
        return {
            "total_probability_mean": None,
            "total_probability_min": None,
            "any_edge_probability_mean": None,
            "any_edge_probability_p95": None,
            "any_edge_probability_max": None,
            "fraction_any_edge_gt_1pct": None,
            "fraction_any_edge_gt_5pct": None,
            "per_mode_edge_probability_mean": [],
            "per_mode_edge_probability_max": [],
        }
    return {
        "total_probability_mean": float(np.mean(total_probability)),
        "total_probability_min": float(np.min(total_probability)),
        "any_edge_probability_mean": float(np.mean(any_edge_probability)),
        "any_edge_probability_p95": float(np.quantile(any_edge_probability, 0.95)),
        "any_edge_probability_max": float(np.max(any_edge_probability)),
        "fraction_any_edge_gt_1pct": float(np.mean(any_edge_probability > 0.01)),
        "fraction_any_edge_gt_5pct": float(np.mean(any_edge_probability > 0.05)),
        "per_mode_edge_probability_mean": [float(x) for x in np.mean(per_mode_edge_probability, axis=0)],
        "per_mode_edge_probability_max": [float(x) for x in np.max(per_mode_edge_probability, axis=0)],
    }

# Loss functions, prediction collection, and dataset evaluation

def zero_truncation_stats_batch(batch_size, n_modes):
    batch_size = int(batch_size)
    return {
        "total_probability": tf.ones((batch_size,), dtype=tf.float32),
        "any_edge_probability": tf.zeros((batch_size,), dtype=tf.float32),
        "per_mode_edge_probability": tf.zeros((batch_size, n_modes), dtype=tf.float32),
    }


def binary_cross_entropy_from_logits(logits, labels):
    labels = tf.cast(labels, tf.float32)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))


def collect_batch_predictions(
    Xb,
    n_modes,
    arch,
    pairs,
    n_layers,
    cutoff,
    enc_vars,
    arch_vars,
    local_vars,
    local_gates,
    clf_head,
    engine,
    encoding_mode,
    observable_readout,
    disable_truncation_stats=False,
):
    logits = []
    total_probabilities = []
    any_edge_probabilities = []
    per_mode_edge_probabilities = []
    for x in Xb:
        probs = run_circuit(
            x,
            n_modes,
            arch,
            pairs,
            n_layers,
            cutoff,
            enc_vars,
            arch_vars,
            local_vars,
            local_gates,
            engine,
            encoding_mode,
        )
        if disable_truncation_stats:
            trunc_stats = zero_truncation_stats_batch(1, n_modes)
            trunc_stats = {
                "total_probability": trunc_stats["total_probability"][0],
                "any_edge_probability": trunc_stats["any_edge_probability"][0],
                "per_mode_edge_probability": trunc_stats["per_mode_edge_probability"][0],
            }
        else:
            trunc_stats = fock_truncation_stats_from_probs(probs, cutoff, n_modes)
        total_probabilities.append(trunc_stats["total_probability"])
        any_edge_probabilities.append(trunc_stats["any_edge_probability"])
        per_mode_edge_probabilities.append(trunc_stats["per_mode_edge_probability"])
        logits.append(photon_logits_from_probs_readout(probs, clf_head, cutoff, n_modes, observable_readout))
    logits = tf.stack(logits)
    probs = tf.math.sigmoid(logits)
    return logits, probs, {
        "total_probability": tf.stack(total_probabilities),
        "any_edge_probability": tf.stack(any_edge_probabilities),
        "per_mode_edge_probability": tf.stack(per_mode_edge_probabilities),
    }


def evaluate_dataset(
    X,
    y,
    batch_size,
    rng,
    n_modes,
    arch,
    pairs,
    n_layers,
    cutoff,
    enc_vars,
    arch_vars,
    local_vars,
    local_gates,
    clf_head,
    engine,
    encoding_mode,
    observable_readout,
    disable_truncation_stats=False,
):
    losses = []
    probs_all = []
    labels_all = []
    total_probability_all = []
    any_edge_probability_all = []
    per_mode_edge_probability_all = []
    for Xb, yb in batches(X, y, batch_size, rng):
        logits, probs, trunc_stats = collect_batch_predictions(
            Xb, n_modes, arch, pairs, n_layers, cutoff,
            enc_vars, arch_vars, local_vars, local_gates, clf_head, engine, encoding_mode, observable_readout,
            disable_truncation_stats,
        )
        loss = binary_cross_entropy_from_logits(logits, yb)
        losses.append(safe_float(loss))
        probs_all.append(probs.numpy())
        labels_all.append(yb)
        total_probability_all.append(trunc_stats["total_probability"].numpy())
        any_edge_probability_all.append(trunc_stats["any_edge_probability"].numpy())
        per_mode_edge_probability_all.append(trunc_stats["per_mode_edge_probability"].numpy())

    probs_full = np.concatenate(probs_all) if probs_all else np.array([], dtype=np.float32)
    labels_full = np.concatenate(labels_all) if labels_all else np.array([], dtype=np.float32)
    total_probability_full = (
        np.concatenate(total_probability_all) if total_probability_all else np.array([], dtype=np.float32)
    )
    any_edge_probability_full = (
        np.concatenate(any_edge_probability_all) if any_edge_probability_all else np.array([], dtype=np.float32)
    )
    per_mode_edge_probability_full = (
        np.concatenate(per_mode_edge_probability_all, axis=0)
        if per_mode_edge_probability_all
        else np.empty((0, n_modes), dtype=np.float32)
    )
    auc = float(roc_auc_score(labels_full, probs_full)) if len(np.unique(labels_full)) >= 2 else 0.5
    acc = float(accuracy_score(labels_full, (probs_full >= 0.5).astype(np.float32))) if len(labels_full) else 0.0
    return {
        "loss": float(np.mean(losses)) if losses else np.inf,
        "auc": auc,
        "acc": acc,
        "probs": probs_full,
        "labels": labels_full,
        "truncation": summarize_truncation_stats(
            total_probability_full, any_edge_probability_full, per_mode_edge_probability_full
        ),
        "truncation_arrays": {
            "total_probability": total_probability_full,
            "any_edge_probability": any_edge_probability_full,
            "per_mode_edge_probability": per_mode_edge_probability_full,
        },
    }


# Plotting and saving training outputs and model weights

def plot_training(history, outdir, run_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].set_title(f"Loss | {run_name}")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(history["train_auc"], label="train AUC")
    axes[1].plot(history["val_auc"], label="val AUC")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("ROC AUC")
    axes[1].set_title(f"AUC | {run_name}")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(outdir / "TRAINING_CURVES.png", dpi=220)
    plt.close(fig)


def plot_roc_curve(labels, scores, outdir, run_name):
    fpr, tpr, thr = roc_curve(labels, scores)
    auc = float(roc_auc_score(labels, scores))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC | {run_name}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "ROC_FINAL.png", dpi=220)
    plt.close(fig)
    return fpr, tpr, thr, auc


def plot_score_hist(labels, scores, outdir, run_name):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(scores[labels == 0], bins=50, alpha=0.65, density=True, label="Background")
    ax.hist(scores[labels == 1], bins=50, alpha=0.65, density=True, label="Signal")
    ax.set_xlabel("Classifier score")
    ax.set_ylabel("Density")
    ax.set_title(f"Score Histogram | {run_name}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "SCORE_HIST_FINAL.png", dpi=220)
    plt.close(fig)


def weight_payload(enc_vars, arch_vars, local_vars, clf_head):
    payload = {}
    groups = {
        "encoder": enc_vars,
        "architecture": arch_vars,
        "local": local_vars,
        "classifier_head": clf_head,
    }
    for group_name, group_vars in groups.items():
        for name, value in group_vars.items():
            if isinstance(value, tf.Variable):
                payload[f"{group_name}__{name}"] = value.numpy()
    return payload

# History to save how params evolved
def init_local_gate_history(local_vars):
    if not local_vars:
        return None
    history = {"epoch": []}
    for name in local_vars:
        history[name] = []
    return history

def record_local_gate_history(local_gate_history, epoch, local_vars):
    if local_gate_history is None or not local_vars:
        return
    local_gate_history["epoch"].append(int(epoch))
    for name, value in local_vars.items():
        local_gate_history[name].append(value.numpy().copy())

def save_local_gate_history(outdir, local_gate_history):
    if local_gate_history is None:
        return None
    payload = {"epoch": np.array(local_gate_history["epoch"], dtype=np.int32)}
    for name, values in local_gate_history.items():
        if name == "epoch":
            continue
        payload[name] = np.array(values, dtype=np.float32)
    history_path = outdir / "local_gate_history.npz"
    np.savez_compressed(history_path, **payload)
    return history_path

def init_epoch_parameter_history(enc_vars, arch_vars, local_vars, clf_head):
    history = {"epoch": []}
    for key in weight_payload(enc_vars, arch_vars, local_vars, clf_head):
        history[key] = []
    return history

def record_epoch_parameter_history(epoch_parameter_history, epoch, enc_vars, arch_vars, local_vars, clf_head):
    if epoch_parameter_history is None:
        return
    epoch_parameter_history["epoch"].append(int(epoch))
    payload = weight_payload(enc_vars, arch_vars, local_vars, clf_head)
    for key, value in payload.items():
        epoch_parameter_history[key].append(np.array(value, copy=True))

def save_epoch_parameter_history(outdir, epoch_parameter_history):
    if epoch_parameter_history is None:
        return None
    payload = {"epoch": np.array(epoch_parameter_history["epoch"], dtype=np.int32)}
    for key, values in epoch_parameter_history.items():
        if key == "epoch":
            continue
        payload[key] = np.array(values)
    history_path = outdir / "epoch_parameter_history.npz"
    np.savez_compressed(history_path, **payload)
    return history_path

def init_step_training_history():
    return {
        "global_step": [],
        "epoch": [],
        "batch_in_epoch": [],
        "loss": [],
    }

def record_step_training_history(step_training_history, global_step, epoch, batch_in_epoch, loss):
    if step_training_history is None:
        return
    step_training_history["global_step"].append(int(global_step))
    step_training_history["epoch"].append(int(epoch))
    step_training_history["batch_in_epoch"].append(int(batch_in_epoch))
    step_training_history["loss"].append(float(loss))

def save_step_training_history(outdir, step_training_history):
    if step_training_history is None:
        return None
    payload = {
        "global_step": np.array(step_training_history["global_step"], dtype=np.int32),
        "epoch": np.array(step_training_history["epoch"], dtype=np.int32),
        "batch_in_epoch": np.array(step_training_history["batch_in_epoch"], dtype=np.int32),
        "loss": np.array(step_training_history["loss"], dtype=np.float32),
    }
    history_path = outdir / "step_training_history.npz"
    np.savez_compressed(history_path, **payload)
    return history_path

def init_step_parameter_history(enc_vars, arch_vars, local_vars, clf_head):
    history = {
        "global_step": [],
        "epoch": [],
        "batch_in_epoch": [],
        "loss": [],
    }
    for key in weight_payload(enc_vars, arch_vars, local_vars, clf_head):
        history[key] = []
    return history

def record_step_parameter_history(
    step_parameter_history,
    global_step,
    epoch,
    batch_in_epoch,
    loss,
    enc_vars,
    arch_vars,
    local_vars,
    clf_head,
):
    if step_parameter_history is None:
        return
    step_parameter_history["global_step"].append(int(global_step))
    step_parameter_history["epoch"].append(int(epoch))
    step_parameter_history["batch_in_epoch"].append(int(batch_in_epoch))
    step_parameter_history["loss"].append(float(loss))
    payload = weight_payload(enc_vars, arch_vars, local_vars, clf_head)
    for key, value in payload.items():
        step_parameter_history[key].append(np.array(value, copy=True))

def save_step_parameter_history(outdir, step_parameter_history):
    if step_parameter_history is None:
        return None
    payload = {
        "global_step": np.array(step_parameter_history["global_step"], dtype=np.int32),
        "epoch": np.array(step_parameter_history["epoch"], dtype=np.int32),
        "batch_in_epoch": np.array(step_parameter_history["batch_in_epoch"], dtype=np.int32),
        "loss": np.array(step_parameter_history["loss"], dtype=np.float32),
    }
    for key, values in step_parameter_history.items():
        if key in {"global_step", "epoch", "batch_in_epoch", "loss"}:
            continue
        payload[key] = np.array(values)
    history_path = outdir / "step_parameter_history.npz"
    np.savez_compressed(history_path, **payload)
    return history_path


def save_trained_model(outdir, run_name, args, n_params, pairs, enc_vars, arch_vars, local_vars, clf_head):
    weights_path = outdir / "trained_model_weights.npz"
    payload = weight_payload(enc_vars, arch_vars, local_vars, clf_head)
    np.savez(weights_path, **payload)

    metadata = {
        "run_name": run_name,
        "arch": args.arch,
        "local_gates": args.local_gates,
        "encoding_mode": args.encoding_mode,
        "encoding_description": encoding_mode_description(args.encoding_mode),
        "observable_readout": args.observable_readout,
        "observable_feature_dim": int(observable_feature_dim(args.modes, args.observable_readout)),
        "modes": int(args.modes),
        "layers": int(args.layers),
        "cutoff": int(args.cutoff),
        "seed": int(args.seed),
        "n_params": int(n_params),
        "pair_list": [list(pair) for pair in pairs],
        "saved_variables": {
            key: list(value.shape) for key, value in payload.items()
        },
        "weights_file": str(weights_path.resolve()),
    }
    with open(outdir / "trained_model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return weights_path



# Main training script

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, required=True)
    ap.add_argument("--val_path", type=str, default=None)
    ap.add_argument("--test_path", type=str, default=None)
    ap.add_argument("--outbase", type=str, default="classifier/runs/sf_entanglement_classifier")
    ap.add_argument("--modes", type=int, required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--arch", type=str, required=True, choices=ARCH_LIST)
    ap.add_argument("--local_gates", type=str, default="srgate", choices=LOCAL_GATES)
    ap.add_argument("--cutoff", type=int, default=CUTOFF)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--n_train_per_class", type=int, default=N_TRAIN_PER_CLASS)
    ap.add_argument("--train_pool_per_class", type=int, default=None)
    ap.add_argument("--n_val_per_class", type=int, default=N_VAL_PER_CLASS)
    ap.add_argument("--n_test_per_class", type=int, default=N_TEST_PER_CLASS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--encoding_mode", type=str, default="trainable_per_mode", choices=ENCODING_MODES)
    ap.add_argument("--observable_readout", type=str, default="mean_photon", choices=OBSERVABLE_READOUTS)
    ap.add_argument("--ck_topology", type=str, default="chain", choices=sorted(TOPOS.keys()), help="Cross-Kerr connectivity for new_entangled_kck")
    ap.add_argument("--no_self_kerr", action="store_true",help="new_entangled_kck: cross-Kerr only (drop the single-mode Kgate)")
    ap.add_argument("--save_epoch_parameter_history", action="store_true")
    ap.add_argument("--save_step_training_history", action="store_true")
    ap.add_argument("--save_step_parameter_history", action="store_true")
    ap.add_argument("--step_parameter_history_stride", type=int, default=1)
    ap.add_argument("--disable_truncation_stats", action="store_true")
    args = ap.parse_args()

    if args.modes < 3 or args.modes > 8:
        raise SystemExit("modes must be in 3..8")
    if args.layers < 1 or args.layers > 3:
        raise SystemExit("layers must be in 1..3")
    if args.step_parameter_history_stride < 1:
        raise SystemExit("step_parameter_history_stride must be >= 1")

    set_seeds(args.seed)
    rng_train = np.random.default_rng(args.seed + 101)
    rng_eval = np.random.default_rng(args.seed + 202)

    if args.arch in {
        "fixed_ring", "trainable_ring",
        "fixed_alltoall", "trainable_alltoall",
        "fixed_chain", "trainable_chain",
        "fixed_star", "trainable_star",
    }:
        topo_key = args.arch.split("_")[1]
        pairs = TOPOS[topo_key](args.modes)
    else:
        pairs = []

    encoding_tag = "" if args.encoding_mode == "trainable_per_mode" else f"_enc{args.encoding_mode}"
    observable_tag = "" if args.observable_readout == "mean_photon" else f"_obs{args.observable_readout}"
    ck_tag = "" if args.arch not in ("new_entangled_kck") else f"_ck{args.ck_topology}{'only' if args.no_self_kerr else ''}"
    run_name = f"clf_{args.arch}_lg{args.local_gates}{encoding_tag}{observable_tag}{ck_tag}_m{args.modes}_L{args.layers}"

    #run_name = f"clf_{args.arch}_lg{args.local_gates}{encoding_tag}{observable_tag}_m{args.modes}_L{args.layers}"
    outdir = (
        Path(args.outbase)
        / f"m{args.modes}"
        / f"L{args.layers}"
        / args.arch
        / f"lg_{args.local_gates}"
    )
    if args.observable_readout != "mean_photon":
        outdir = outdir / f"obs_{args.observable_readout}"
    #if args.cutoff != CUTOFF:
    outdir = outdir / f"cutoff_{args.cutoff}"

    if args.arch in ("new_entangled_kck"):
        self_tag = "ckonly" if args.no_self_kerr else "selfkerr"
        outdir = outdir / f"ck_{args.ck_topology}_{self_tag}"
    outdir = outdir / f"seed{args.seed}"

    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Run         : {run_name}")
    print(f"Data        : {args.path}")
    print(f"Modes       : {args.modes}")
    print(f"Layers      : {args.layers}")
    print(f"Arch        : {args.arch}")
    print(f"Local gates : {args.local_gates}")
    print(f"Encoding    : {args.encoding_mode}")
    print(f"Observable  : {args.observable_readout}")
    print(f"Trunc stats : {not bool(args.disable_truncation_stats)}")
    print(f"Save epoch params: {bool(args.save_epoch_parameter_history)}")
    print(f"Save step loss: {bool(args.save_step_training_history)}")
    print(f"Save step params: {bool(args.save_step_parameter_history)}")
    if args.save_step_parameter_history:
        print(f"Step param stride: {args.step_parameter_history_stride}")
    print(f"Cutoff      : {args.cutoff}")
    print(f"Epochs      : {args.epochs}")
    print(f"Batch       : {args.batch}")
    print(f"LR          : {args.lr}")
    print(f"Train/class : {args.n_train_per_class}")
    if args.train_pool_per_class is not None:
        print(f"Train pool  : {args.train_pool_per_class}")
    print(f"Val/class   : {args.n_val_per_class}")
    print(f"Test/class  : {args.n_test_per_class}")
    print(f"Seed        : {args.seed}")
    print("=" * 72)

    if (args.val_path is None) ^ (args.test_path is None):
        raise SystemExit("Provide both --val_path and --test_path together, or neither.")

    if args.val_path and args.test_path:
        train_const, train_truth, train_jet_pt = load_data(args.path)
        val_const, val_truth, val_jet_pt = load_data(args.val_path)
        test_const, test_truth, test_jet_pt = load_data(args.test_path)
        data = prepare_balanced_splits_from_files(
            train_const,
            train_truth,
            train_jet_pt,
            val_const,
            val_truth,
            val_jet_pt,
            test_const,
            test_truth,
            test_jet_pt,
            args.modes,
            args.n_train_per_class,
            args.n_val_per_class,
            args.n_test_per_class,
            args.seed,
        )
    else:
        const, truth, jet_pt = load_data(args.path)
        data = prepare_balanced_splits(
            const, truth, jet_pt, args.modes,
            args.n_train_per_class, args.n_val_per_class, args.n_test_per_class,
            args.seed, args.train_pool_per_class,
        )

    enc_vars = init_encoder_vars(args.modes, args.seed, args.encoding_mode)
    arch_vars = init_arch_vars(args.arch, pairs, args.modes, args.layers, args.seed, ck_topology=args.ck_topology,
        use_self_kerr=not args.no_self_kerr)
    local_vars = init_local_vars(args.modes, args.layers, args.local_gates, args.seed)
    clf_head = init_classifier_head_for_readout(args.modes, args.seed, args.observable_readout)

    train_vars = list(enc_vars.values()) + list(clf_head.values())
    if args.arch.startswith("trainable_") or args.arch in {"new_entangled", "new_entangled_k", "new_entangled_kck", "new_entangled_k0", "new_circuit_like"}:
        for value in arch_vars.values():
            if isinstance(value, tf.Variable):
                train_vars.append(value)
    if local_vars:
        train_vars += list(local_vars.values())

    n_params = sum(int(np.prod(v.shape)) for v in train_vars)
    print(f"Trainable params: {n_params}")

    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    backend_mode = "batched" if supports_batched_backend(args.arch) else "single"
    print(f"Backend     : {backend_mode}")

    if backend_mode == "batched":
        batched_prog = build_batched_program(args.modes, args.arch, pairs, args.layers, args.local_gates)
        engine_cache = {}

        def get_batched_engine(batch_size):
            batch_size = int(batch_size)
            if batch_size not in engine_cache:
                engine_cache[batch_size] = sf.Engine(
                    "tf",
                    backend_options={"cutoff_dim": int(args.cutoff), "batch_size": batch_size},
                )
            return engine_cache[batch_size]

        def run_batched_probs_backend(Xb):
            Xb = np.asarray(Xb, dtype=np.float32)
            engine = get_batched_engine(len(Xb))
            state = engine.run(
                batched_prog,
                args=build_batched_feed(
                    Xb,
                    args.modes,
                    args.arch,
                    pairs,
                    args.layers,
                    enc_vars,
                    arch_vars,
                    local_vars,
                    args.local_gates,
                    args.encoding_mode,
                ),
            ).state
            probs = state.all_fock_probs()
            engine.reset()
            return probs

        def collect_logits_backend(Xb):
            probs = run_batched_probs_backend(Xb)
            return photon_logits_from_probs_batch_readout(
                probs, clf_head, args.cutoff, args.modes, args.observable_readout
            )

        def collect_predictions_backend(Xb):
            probs = run_batched_probs_backend(Xb)
            logits = photon_logits_from_probs_batch_readout(
                probs, clf_head, args.cutoff, args.modes, args.observable_readout
            )
            scores = tf.math.sigmoid(logits)
            if args.disable_truncation_stats:
                trunc_stats = zero_truncation_stats_batch(len(Xb), args.modes)
            else:
                # Truncation diagnostics are for reporting only; detach them so
                # TensorFlow never needs to backpropagate through high-rank
                # gathers/reductions when the training loss only depends on logits.
                trunc_stats = fock_truncation_stats_from_probs_batch(
                    tf.stop_gradient(probs), args.cutoff, args.modes
                )
            return logits, scores, trunc_stats

        def evaluate_dataset_backend(X, y, batch_size):
            losses = []
            probs_all = []
            labels_all = []
            total_probability_all = []
            any_edge_probability_all = []
            per_mode_edge_probability_all = []
            for Xb, yb in sequential_batches(X, y, batch_size):
                logits, probs, trunc_stats = collect_predictions_backend(Xb)
                loss = binary_cross_entropy_from_logits(logits, yb)
                losses.append(safe_float(loss))
                probs_all.append(probs.numpy())
                labels_all.append(yb)
                total_probability_all.append(trunc_stats["total_probability"].numpy())
                any_edge_probability_all.append(trunc_stats["any_edge_probability"].numpy())
                per_mode_edge_probability_all.append(trunc_stats["per_mode_edge_probability"].numpy())

            probs_full = np.concatenate(probs_all) if probs_all else np.array([], dtype=np.float32)
            labels_full = np.concatenate(labels_all) if labels_all else np.array([], dtype=np.float32)
            total_probability_full = (
                np.concatenate(total_probability_all) if total_probability_all else np.array([], dtype=np.float32)
            )
            any_edge_probability_full = (
                np.concatenate(any_edge_probability_all) if any_edge_probability_all else np.array([], dtype=np.float32)
            )
            per_mode_edge_probability_full = (
                np.concatenate(per_mode_edge_probability_all, axis=0)
                if per_mode_edge_probability_all
                else np.empty((0, args.modes), dtype=np.float32)
            )
            auc = float(roc_auc_score(labels_full, probs_full)) if len(np.unique(labels_full)) >= 2 else 0.5
            acc = float(accuracy_score(labels_full, (probs_full >= 0.5).astype(np.float32))) if len(labels_full) else 0.0
            return {
                "loss": float(np.mean(losses)) if losses else np.inf,
                "auc": auc,
                "acc": acc,
                "probs": probs_full,
                "labels": labels_full,
                "truncation": summarize_truncation_stats(
                    total_probability_full, any_edge_probability_full, per_mode_edge_probability_full
                ),
                "truncation_arrays": {
                    "total_probability": total_probability_full,
                    "any_edge_probability": any_edge_probability_full,
                    "per_mode_edge_probability": per_mode_edge_probability_full,
                },
            }
    else:
        engine = sf.Engine("tf", backend_options={"cutoff_dim": int(args.cutoff)})

        def collect_logits_backend(Xb):
            return collect_batch_predictions(
                Xb,
                args.modes,
                args.arch,
                pairs,
                args.layers,
                args.cutoff,
                enc_vars,
                arch_vars,
                local_vars,
                args.local_gates,
                clf_head,
                engine,
                args.encoding_mode,
                args.observable_readout,
                True,
            )[0]

        def collect_predictions_backend(Xb):
            return collect_batch_predictions(
                Xb,
                args.modes,
                args.arch,
                pairs,
                args.layers,
                args.cutoff,
                enc_vars,
                arch_vars,
                local_vars,
                args.local_gates,
                clf_head,
                engine,
                args.encoding_mode,
                args.observable_readout,
            )

        def evaluate_dataset_backend(X, y, batch_size):
            return evaluate_dataset(
                X,
                y,
                batch_size,
                rng_eval,
                args.modes,
                args.arch,
                pairs,
                args.layers,
                args.cutoff,
                enc_vars,
                arch_vars,
                local_vars,
                args.local_gates,
                clf_head,
                engine,
                args.encoding_mode,
                args.observable_readout,
                args.disable_truncation_stats,
            )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_auc": [],
        "val_auc": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
        "train_truncation_total_probability_mean": [],
        "val_truncation_total_probability_mean": [],
        "train_truncation_any_edge_probability_mean": [],
        "val_truncation_any_edge_probability_mean": [],
        "train_truncation_any_edge_probability_p95": [],
        "val_truncation_any_edge_probability_p95": [],
        "train_truncation_fraction_any_edge_gt_1pct": [],
        "val_truncation_fraction_any_edge_gt_1pct": [],
        "train_truncation_fraction_any_edge_gt_5pct": [],
        "val_truncation_fraction_any_edge_gt_5pct": [],
        "epoch_seconds": [],
    }

    best_val = np.inf #np.inf
    best_epoch = -1
    best_weights = None
    bad_epochs = 0
    lr_bad_epochs = 0

    t0 = time.time()
    local_gate_history = init_local_gate_history(local_vars)
    epoch_parameter_history = (
        init_epoch_parameter_history(enc_vars, arch_vars, local_vars, clf_head)
        if args.save_epoch_parameter_history
        else None
    )
    epoch_parameter_history_path = None
    step_training_history = init_step_training_history() if args.save_step_training_history else None
    step_training_history_path = None
    step_parameter_history = (
        init_step_parameter_history(enc_vars, arch_vars, local_vars, clf_head)
        if args.save_step_parameter_history
        else None
    )
    step_parameter_history_path = None
    global_step = 0

    for epoch in tqdm(range(1, args.epochs + 1), desc=run_name):
        ep_start = time.time()

        for batch_in_epoch, (Xb, yb) in enumerate(
            tqdm(batches(data["X_train"], data["y_train"], args.batch, rng_train), leave=False, desc=f"ep{epoch}"),
            start=1,
        ):
            with tf.GradientTape() as tape:
                logits = collect_logits_backend(Xb)
                loss = binary_cross_entropy_from_logits(logits, yb)
            grads = tape.gradient(loss, train_vars)
            opt.apply_gradients([(g, v) for g, v in zip(grads, train_vars) if g is not None])
            global_step += 1
            step_loss = safe_float(loss)
            if args.save_step_training_history:
                record_step_training_history(step_training_history, global_step, epoch, batch_in_epoch, step_loss)
            if args.save_step_parameter_history and (global_step % args.step_parameter_history_stride == 0):
                record_step_parameter_history(
                    step_parameter_history,
                    global_step,
                    epoch,
                    batch_in_epoch,
                    step_loss,
                    enc_vars,
                    arch_vars,
                    local_vars,
                    clf_head,
                )

        train_metrics = evaluate_dataset_backend(data["X_train"], data["y_train"], args.batch)
        val_metrics = evaluate_dataset_backend(data["X_val"], data["y_val"], args.batch)

        epoch_seconds = time.time() - ep_start
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_auc"].append(train_metrics["auc"])
        history["val_auc"].append(val_metrics["auc"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])
        history["train_truncation_total_probability_mean"].append(
            train_metrics["truncation"]["total_probability_mean"]
        )
        history["val_truncation_total_probability_mean"].append(
            val_metrics["truncation"]["total_probability_mean"]
        )
        history["train_truncation_any_edge_probability_mean"].append(
            train_metrics["truncation"]["any_edge_probability_mean"]
        )
        history["val_truncation_any_edge_probability_mean"].append(
            val_metrics["truncation"]["any_edge_probability_mean"]
        )
        history["train_truncation_any_edge_probability_p95"].append(
            train_metrics["truncation"]["any_edge_probability_p95"]
        )
        history["val_truncation_any_edge_probability_p95"].append(
            val_metrics["truncation"]["any_edge_probability_p95"]
        )
        history["train_truncation_fraction_any_edge_gt_1pct"].append(
            train_metrics["truncation"]["fraction_any_edge_gt_1pct"]
        )
        history["val_truncation_fraction_any_edge_gt_1pct"].append(
            val_metrics["truncation"]["fraction_any_edge_gt_1pct"]
        )
        history["train_truncation_fraction_any_edge_gt_5pct"].append(
            train_metrics["truncation"]["fraction_any_edge_gt_5pct"]
        )
        history["val_truncation_fraction_any_edge_gt_5pct"].append(
            val_metrics["truncation"]["fraction_any_edge_gt_5pct"]
        )
        history["epoch_seconds"].append(epoch_seconds)
        history["lr"].append(float(opt.learning_rate.numpy()))

        record_local_gate_history(local_gate_history, epoch, local_vars)
        local_history_path = save_local_gate_history(outdir, local_gate_history)
        if args.save_epoch_parameter_history:
            record_epoch_parameter_history(epoch_parameter_history, epoch, enc_vars, arch_vars, local_vars, clf_head)
            epoch_parameter_history_path = save_epoch_parameter_history(outdir, epoch_parameter_history)
        if args.save_step_training_history:
            step_training_history_path = save_step_training_history(outdir, step_training_history)
        if args.save_step_parameter_history:
            step_parameter_history_path = save_step_parameter_history(outdir, step_parameter_history)

        tqdm.write(
            f"[{run_name}] ep{epoch:02d} | "
            f"tr_loss={train_metrics['loss']:.5f} val_loss={val_metrics['loss']:.5f} | "
            f"tr_auc={train_metrics['auc']:.4f} val_auc={val_metrics['auc']:.4f} | "
            f"tr_acc={train_metrics['acc']:.4f} val_acc={val_metrics['acc']:.4f} | "
            f"val_edge={val_metrics['truncation']['any_edge_probability_mean']:.4f}"
        )

        with open(outdir / "live_history.json", "w") as f:
            json.dump(
                {
                    "run_name": run_name,
                    "arch": args.arch,
                    "local_gates": args.local_gates,
                    "encoding_mode": args.encoding_mode,
                    "encoding_description": encoding_mode_description(args.encoding_mode),
                    "observable_readout": args.observable_readout,
                    "observable_feature_dim": int(observable_feature_dim(args.modes, args.observable_readout)),
                    "modes": args.modes,
                    "layers": args.layers,
                    "cutoff": args.cutoff,
                    "n_params": n_params,
                    "backend_mode": backend_mode,
                    "current_epoch": epoch,
                    "global_step": global_step,
                    "best_epoch": best_epoch,
                    "best_val_metric": None if not np.isfinite(best_val) else float(best_val),
                    "local_gate_history_file": str(local_history_path.resolve()) if local_history_path else None,
                    "epoch_parameter_history_file": (
                        str(epoch_parameter_history_path.resolve()) if epoch_parameter_history_path else None
                    ),
                    "step_training_history_file": (
                        str(step_training_history_path.resolve()) if step_training_history_path else None
                    ),
                    "step_parameter_history_file": (
                        str(step_parameter_history_path.resolve()) if step_parameter_history_path else None
                    ),
                    "step_parameter_history_stride": (
                        int(args.step_parameter_history_stride) if args.save_step_parameter_history else None
                    ),
                    "history": history,
                },
                f,
                indent=2,
            )

        plot_training(history, outdir, run_name)


        # Early stopping CHANGE to validation based

        if val_metrics["loss"] < best_val - MIN_DELTA:
            best_val = float(val_metrics["loss"])
            best_epoch = epoch
            bad_epochs = 0
            lr_bad_epochs = 0
            best_weights = [v.numpy().copy() for v in train_vars]
        else:
            bad_epochs += 1
            lr_bad_epochs += 1

            # LR decay on plateau
            if lr_bad_epochs >= LR_PATIENCE:
                current_lr = float(opt.learning_rate.numpy())
                new_lr = max(current_lr * LR_FACTOR, MIN_LR)
                if new_lr < current_lr:
                    opt.learning_rate.assign(new_lr)
                    tqdm.write(f"[{run_name}] LR {current_lr:.5g} -> {new_lr:.5g}")
                lr_bad_epochs = 0

            if bad_epochs >= PATIENCE:
                tqdm.write(f"[{run_name}] Early stop at epoch {epoch} (best={best_epoch})")
                break

        #if val_metrics["auc"] > best_val + MIN_DELTA:
        #    best_val = float(val_metrics["auc"])
        #    best_epoch = epoch
        #    bad_epochs = 0
        #    lr_bad_epochs = 0
        #    best_weights = [v.numpy().copy() for v in train_vars]
        #else:
        #    bad_epochs += 1
        #    lr_bad_epochs += 1

            # LR decay on plateau
        #    if lr_bad_epochs >= LR_PATIENCE:
        #        current_lr = float(opt.learning_rate.numpy())
        #        new_lr = max(current_lr * LR_FACTOR, MIN_LR)
        #        if new_lr < current_lr:
        #            opt.learning_rate.assign(new_lr)
        #            tqdm.write(f"[{run_name}] LR {current_lr:.5g} -> {new_lr:.5g}")
        #        lr_bad_epochs = 0

        #    if bad_epochs >= PATIENCE:
        #        tqdm.write(f"[{run_name}] Early stop at epoch {epoch} (best={best_epoch})")
        #        break

        
        #if val_metrics["loss"] < best_val - MIN_DELTA:
        #    best_val = float(val_metrics["loss"])
        #    best_epoch = epoch
        #    bad_epochs = 0
        #    best_weights = [v.numpy().copy() for v in train_vars]
        #else:
        #    bad_epochs += 1
        #    if bad_epochs >= PATIENCE:
        #        tqdm.write(f"[{run_name}] Early stop at epoch {epoch} (best={best_epoch})")
        #        break

    if best_weights is not None:
        for var, weight in zip(train_vars, best_weights):
            var.assign(weight)

    total_seconds = time.time() - t0
    test_metrics = evaluate_dataset_backend(data["X_test"], data["y_test"], args.batch)

    fpr, tpr, thr, auc = plot_roc_curve(test_metrics["labels"], test_metrics["probs"], outdir, run_name)
    plot_score_hist(test_metrics["labels"], test_metrics["probs"], outdir, run_name)

    np.savez(
        outdir / "scores_labels.npz",
        y_test=test_metrics["labels"],
        scores=test_metrics["probs"],
        fpr=fpr,
        tpr=tpr,
        thr=thr,
        auc=np.array([auc], dtype=np.float32),
        truncation_total_probability=test_metrics["truncation_arrays"]["total_probability"],
        truncation_any_edge_probability=test_metrics["truncation_arrays"]["any_edge_probability"],
        truncation_per_mode_edge_probability=test_metrics["truncation_arrays"]["per_mode_edge_probability"],
    )
    np.savez(
        outdir / "training_history.npz",
        **{k: np.array(v, dtype=np.float32) for k, v in history.items()},
    )
    weights_path = save_trained_model(
        outdir,
        run_name,
        args,
        n_params,
        pairs,
        enc_vars,
        arch_vars,
        local_vars,
        clf_head,
    )
    local_history_path = save_local_gate_history(outdir, local_gate_history)
    if args.save_epoch_parameter_history:
        epoch_parameter_history_path = save_epoch_parameter_history(outdir, epoch_parameter_history)
    if args.save_step_training_history:
        step_training_history_path = save_step_training_history(outdir, step_training_history)
    if args.save_step_parameter_history:
        step_parameter_history_path = save_step_parameter_history(outdir, step_parameter_history)

    summary = {
        "run_name": run_name,
        "arch": args.arch,
        "local_gates": args.local_gates,
        "encoding_mode": args.encoding_mode,
        "encoding_description": encoding_mode_description(args.encoding_mode),
        "observable_readout": args.observable_readout,
        "observable_feature_dim": int(observable_feature_dim(args.modes, args.observable_readout)),
        "modes": args.modes,
        "layers": args.layers,
        "cutoff": args.cutoff,
        "backend_mode": backend_mode,
        "epochs_requested": args.epochs,
        "best_epoch": int(best_epoch),
        "best_val_metric": None if not np.isfinite(best_val) else float(best_val),
        "best_val_auc": float(history["val_auc"][best_epoch - 1]) if best_epoch > 0 else None,
        "best_train_auc": float(history["train_auc"][best_epoch - 1]) if best_epoch > 0 else None,
        "test_auc": float(auc),
        "test_loss": float(test_metrics["loss"]),
        "test_acc": float(test_metrics["acc"]),
        "test_truncation": test_metrics["truncation"],
        "best_train_truncation": (
            {
                key: history[f"train_truncation_{key}"][best_epoch - 1]
                for key in [
                    "total_probability_mean",
                    "any_edge_probability_mean",
                    "any_edge_probability_p95",
                    "fraction_any_edge_gt_1pct",
                    "fraction_any_edge_gt_5pct",
                ]
            }
            if best_epoch > 0
            else None
        ),
        "best_val_truncation": (
            {
                key: history[f"val_truncation_{key}"][best_epoch - 1]
                for key in [
                    "total_probability_mean",
                    "any_edge_probability_mean",
                    "any_edge_probability_p95",
                    "fraction_any_edge_gt_1pct",
                    "fraction_any_edge_gt_5pct",
                ]
            }
            if best_epoch > 0
            else None
        ),
        "n_params": int(n_params),
        "n_train": int(len(data["X_train"])),
        "n_train_per_class": int(args.n_train_per_class),
        "n_train_pool_per_class": int(data["n_train_pool_per_class"]),
        "n_val": int(len(data["X_val"])),
        "n_val_per_class": int(args.n_val_per_class),
        "n_test": int(len(data["X_test"])),
        "n_test_per_class": int(args.n_test_per_class),
        "seed": int(args.seed),
        "train_path": str(args.path),
        "val_path": str(args.val_path) if args.val_path else None,
        "test_path": str(args.test_path) if args.test_path else None,
        "background_label": int(data["bkg_label"]),
        "signal_label": int(data["sig_label"]),
        "total_seconds": float(total_seconds),
        "output_dir": str(outdir.resolve()),
        "weights_file": str(weights_path.resolve()),
        "local_gate_history_file": str(local_history_path.resolve()) if local_history_path else None,
        "epoch_parameter_history_file": (
            str(epoch_parameter_history_path.resolve()) if epoch_parameter_history_path else None
        ),
        "step_training_history_file": (
            str(step_training_history_path.resolve()) if step_training_history_path else None
        ),
        "step_parameter_history_file": (
            str(step_parameter_history_path.resolve()) if step_parameter_history_path else None
        ),
        "step_parameter_history_stride": (
            int(args.step_parameter_history_stride) if args.save_step_parameter_history else None
        ),
    }
    with open(outdir / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{run_name}] DONE | test_auc={auc:.4f} | {outdir}")


if __name__ == "__main__":
    main()

