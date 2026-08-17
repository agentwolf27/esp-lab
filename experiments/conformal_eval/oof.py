"""Stage 1: out-of-fold (leave-one-group-out) probabilities from cached embeddings.

No audio, no encoders, no torch.  Reads out_cross/{emb_wavlm.npy,meta.json} and
out_pigs/{emb_wavlm.npy,meta.json}, both of which were written together so the
row order is guaranteed to match `cross_species.load_all()` / `load_pigs()`.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import json
import time

import numpy as np

CTX = "/private/tmp/claude-501/-Volumes-SSD/eb358682-62a6-4aab-8b76-0d59bcd47add/scratchpad/ctx"
OUT = os.path.join(CTX, "out_conformal")
os.makedirs(OUT, exist_ok=True)
LAYERS = (9, 6)


def zgroup(X, g):
    """z-score within group (pigs_run.zgroup / cross_species.zscore_within)."""
    Z = X.copy()
    for s in np.unique(g):
        m = g == s
        Z[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-8)
    return Z


def fit(Xa, ya):
    """the project's probe (pigs_run.fit)."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=3000, C=1.0,
                              class_weight="balanced").fit(Xa, ya)


def load_layer(layer):
    """-> dict name -> (X z-scored, y, group)."""
    mc = json.load(open(os.path.join(CTX, "out_cross", "meta.json")))
    Ec = np.load(os.path.join(CTX, "out_cross", "emb_wavlm.npy"), mmap_mode="r")
    Xc = np.asarray(Ec[layer], dtype=np.float32)
    sp = np.array([r["species"] for r in mc])
    yc = np.array([r["pol"] for r in mc], dtype=int)
    ic = np.array([r["indiv"] for r in mc])
    ctxc = np.array([r["ctx"] for r in mc])
    # z-score WITHIN species, exactly as cross_species.zscore_within does
    Zc = zgroup(Xc, sp)

    mp = json.load(open(os.path.join(CTX, "out_pigs", "meta.json")))
    Ep = np.load(os.path.join(CTX, "out_pigs", "emb_wavlm.npy"), mmap_mode="r")
    Xp = np.asarray(Ep[layer], dtype=np.float32)
    yp = np.array([r["pol"] for r in mp], dtype=int)
    tm = np.array([r["team"] for r in mp])
    ctxp = np.array([r["ctx"] for r in mp])
    # pigs: plain global z-score, as pigs_run part A does
    Zp = zgroup(Xp, np.zeros(len(yp)))

    return {
        "cats": dict(X=Zc[sp == "cat"], y=yc[sp == "cat"], g=ic[sp == "cat"],
                     ctx=ctxc[sp == "cat"], unit="individual"),
        "dogs": dict(X=Zc[sp == "dog"], y=yc[sp == "dog"], g=ic[sp == "dog"],
                     ctx=ctxc[sp == "dog"], unit="individual"),
        "pigs": dict(X=Zp, y=yp, g=tm, ctx=ctxp, unit="lab"),
    }


def logo_oof(X, y, g):
    """leave-one-group-out P(y=1); every clip scored by a probe that never saw
    its own individual / lab."""
    p1 = np.full(len(y), np.nan)
    for grp in np.unique(g):
        te = g == grp
        clf = fit(X[~te], y[~te])
        p1[te] = clf.predict_proba(X[te])[:, list(clf.classes_).index(1)]
    assert np.isfinite(p1).all()
    return p1


def main():
    for layer in LAYERS:
        cache = os.path.join(OUT, f"oof_L{layer}.npz")
        if os.path.exists(cache):
            print(f"L{layer}: cached"); continue
        t0 = time.time()
        D = load_layer(layer)
        store = {}
        for name, d in D.items():
            t1 = time.time()
            p1 = logo_oof(d["X"], d["y"], d["g"])
            acc = float(((p1 >= .5).astype(int) == d["y"]).mean())
            print(f"  L{layer} {name:5s} n={len(d['y']):5d} groups={len(set(d['g'])):3d} "
                  f"oof_acc={acc:.3f}  ({time.time()-t1:.1f}s)", flush=True)
            store[f"{name}_p1"] = p1
            store[f"{name}_y"] = d["y"]
            store[f"{name}_g"] = d["g"]
            store[f"{name}_ctx"] = d["ctx"]
        np.savez(cache, **store)
        print(f"L{layer} done in {time.time()-t0:.1f}s -> {cache}", flush=True)

    # --- extra scores needed for the cross-species SHIFT case (task 5).
    # Single fixed probe per split: train on a subset of source groups, calibrate
    # on the held-out source groups, test on the whole target species.
    for layer in LAYERS:
        cache = os.path.join(OUT, f"shift_L{layer}.npz")
        if os.path.exists(cache):
            print(f"shift L{layer}: cached"); continue
        D = load_layer(layer)
        rng = np.random.default_rng(0)
        store = {}
        for src, tgt in (("dogs", "cats"), ("cats", "dogs")):
            S, T = D[src], D[tgt]
            gs = np.unique(S["g"])
            R = 20
            cal_p = np.full((R, len(S["y"])), np.nan)   # source cal-group scores
            tgt_p = np.full((R, len(T["y"])), np.nan)   # target scores
            cal_mask = np.zeros((R, len(S["y"])), bool)
            for r in range(R):
                perm = rng.permutation(gs)
                k = len(gs) // 2
                tr_g, cal_g = perm[:len(gs) - k], perm[len(gs) - k:]
                tr = np.isin(S["g"], tr_g)
                if len(np.unique(S["y"][tr])) < 2:
                    continue
                clf = fit(S["X"][tr], S["y"][tr])
                j = list(clf.classes_).index(1)
                cal_p[r] = clf.predict_proba(S["X"])[:, j]
                cal_mask[r] = ~tr
                tgt_p[r] = clf.predict_proba(T["X"])[:, j]
            store[f"{src}2{tgt}_calp"] = cal_p
            store[f"{src}2{tgt}_calmask"] = cal_mask
            store[f"{src}2{tgt}_tgtp"] = tgt_p
            store[f"{src}2{tgt}_srcy"] = S["y"]
            store[f"{src}2{tgt}_tgty"] = T["y"]
            store[f"{src}2{tgt}_tgtg"] = T["g"]
            print(f"  shift L{layer} {src}->{tgt} done", flush=True)
        np.savez(cache, **store)


if __name__ == "__main__":
    main()
