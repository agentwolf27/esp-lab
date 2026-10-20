"""Shared loading / math for the out_sae analysis. Read-only w.r.t. existing out_*/."""
from __future__ import annotations

import json, os
import numpy as np

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(D, "out_sae")
ACO = ["log_dur", "log_energy", "centroid", "zcr"]


def zgroup(X, g):
    """z-score within group (matches pigs_run.zgroup)."""
    Z = X.copy()
    for s in np.unique(g):
        m = g == s
        Z[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-8)
    return Z


def load(layers):
    """Return dict species -> dict with X (per requested layer), y, grp (null unit),
    zgrp (z-scoring unit), A (acoustics). Embeddings are read via mmap and only the
    requested layers are materialised, to keep peak memory ~tens of MB."""
    meta_cd = json.load(open(os.path.join(D, "out_cross", "meta.json")))
    meta_pg = json.load(open(os.path.join(D, "out_pigs", "meta.json")))
    aco = np.load(os.path.join(OUT, "acoustics.npz"))
    Ecd = np.load(os.path.join(D, "out_cross", "emb_wavlm.npy"), mmap_mode="r")
    Epg = np.load(os.path.join(D, "out_pigs", "emb_wavlm.npy"), mmap_mode="r")

    sp = np.array([r["species"] for r in meta_cd])
    y_cd = np.array([r["pol"] for r in meta_cd])
    ind = np.array([r["indiv"] for r in meta_cd])
    ctx_cd = np.array([r["ctx"] for r in meta_cd])
    A_cd = aco["catdog"]

    out = {}
    for s in ["cat", "dog"]:
        k = sp == s
        out[s] = {
            "X": {li: np.asarray(Ecd[li][k], dtype=np.float32) for li in layers},
            "y": y_cd[k], "grp": ind[k],                      # permutation unit = individual
            "zgrp": np.zeros(int(k.sum()), dtype=int),        # z-score over the whole species
            "A": A_cd[k], "ctx": ctx_cd[k], "n": int(k.sum()),
        }
    team = np.array([m["team"] for m in meta_pg])
    out["pig"] = {
        "X": {li: np.asarray(Epg[li], dtype=np.float32) for li in layers},
        "y": np.array([m["pol"] for m in meta_pg]),
        "grp": team,                                          # permutation unit = team
        "zgrp": team,                                         # z-score within team
        "A": aco["pig"], "ctx": np.array([m["ctx"] for m in meta_pg]), "n": len(meta_pg),
    }
    del Ecd, Epg
    return out


def unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def affect_dir(Z, y):
    """normalize(mean_neg - mean_pos)"""
    return unit(Z[y == 1].mean(0) - Z[y == 0].mean(0))


def duration_dir(Z, logdur):
    """OLS coefficient vector of each embedding dim on log-duration (+intercept)."""
    d = logdur - logdur.mean()
    beta = (d[:, None] * Z).sum(0) / ((d ** 2).sum() + 1e-12)
    return beta


def residualise(Z, logdur):
    """Regress log-duration out of every dimension (per species)."""
    d = logdur - logdur.mean()
    beta = duration_dir(Z, logdur)
    return Z - d[:, None] * beta[None, :]


def perm_within(y, g, rng):
    yp = y.copy()
    for s in np.unique(g):
        m = g == s
        yp[m] = rng.permutation(y[m])
    return yp
