"""
PHASE 1 of the encoder plan, on cached embeddings. No GPU, no retraining.

1a. FUSION. eGeMAPS-88 and WavLM-768 score about the same honestly. Do they make
    DIFFERENT errors? If so, concatenating should beat either. Never tested.

1b. IDENTITY-SUBSPACE REMOVAL (the headline experiment). Estimate the identity
    directions from the TRAINING animals only, project them out of the embedding,
    and re-run the honest context probe.

    Iterative Nullspace Projection (INLP, Ravfogel et al.): repeatedly fit a linear
    classifier that predicts identity, then project onto the nullspace of its weights.
    Everything is fit on training groups only and applied to the held-out group, so
    no test information leaks into the projection.

    THE EVALUATION TRAP: you can make identity undecodable by destroying the
    representation. So we report BOTH numbers at every rank:
        context accuracy (held-out group)  -- must hold
        identity accuracy (train groups)   -- must fall
    A win is identity down AND context flat-or-up. Both down is mere compression.
    We also report a RANDOM-DIRECTION control: project out the same number of random
    directions. If random projection does as well, INLP is doing nothing special.
"""
from __future__ import annotations

import json, os, warnings
import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "2")
D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "out_invariance"); os.makedirs(OUT, exist_ok=True)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import StratifiedKFold


def zsc(tr, te):
    m, s = tr.mean(0), tr.std(0) + 1e-8
    return (tr - m) / s, (te - m) / s


def ctx_loGo(X, y, grp):
    """leave-one-group-out balanced accuracy"""
    yp = np.empty_like(y)
    for g in np.unique(grp):
        te = grp == g
        if len(np.unique(y[~te])) < 2:
            yp[te] = y[~te][0]; continue
        a, b = zsc(X[~te], X[te])
        yp[te] = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(a, y[~te]).predict(b)
    return bacc(y, yp)


def ident_acc(X, grp, min_n=5):
    ok = np.isin(grp, [g for g in np.unique(grp) if (grp == g).sum() >= min_n])
    Xi, gi = X[ok], grp[ok]
    if len(np.unique(gi)) < 2:
        return float("nan")
    yp = np.empty(len(gi), dtype=object)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xi, gi):
        a, b = zsc(Xi[tr], Xi[te])
        yp[te] = LogisticRegression(max_iter=3000, C=1.0).fit(a, gi[tr]).predict(b)
    return float((yp == gi).mean())


def inlp_directions(X, grp, n_dir, seed=0):
    """Return an orthonormal basis of the identity subspace, INLP-style.
    X, grp are TRAINING data only."""
    Xw = X.copy()
    dirs = []
    for _ in range(n_dir):
        if len(np.unique(grp)) < 2:
            break
        clf = LogisticRegression(max_iter=1500, C=1.0, random_state=seed).fit(
            (Xw - Xw.mean(0)) / (Xw.std(0) + 1e-8), grp)
        W = np.atleast_2d(clf.coef_)
        for w in W:
            w = w - sum(np.dot(w, d) * d for d in dirs)
            n = np.linalg.norm(w)
            if n < 1e-8:
                continue
            d = w / n
            dirs.append(d)
            Xw = Xw - np.outer(Xw @ d, d)
            if len(dirs) >= n_dir:
                break
    return np.array(dirs) if dirs else np.zeros((0, X.shape[1]))


def project_out(X, dirs):
    for d in dirs:
        X = X - np.outer(X @ d, d)
    return X


def loGo_with_projection(X, y, grp, n_dir, mode="inlp", seed=0):
    """Honest: the projection is estimated on training groups only, per fold."""
    rng = np.random.default_rng(seed)
    yp = np.empty_like(y)
    for g in np.unique(grp):
        te = grp == g
        if len(np.unique(y[~te])) < 2:
            yp[te] = y[~te][0]; continue
        Xtr, Xte = X[~te], X[te]
        if n_dir > 0:
            if mode == "inlp":
                dirs = inlp_directions(Xtr, grp[~te], n_dir, seed)
            else:                                   # random-direction control
                A = rng.standard_normal((n_dir, X.shape[1]))
                dirs = np.linalg.qr(A.T)[0].T
            Xtr, Xte = project_out(Xtr, dirs), project_out(Xte, dirs)
        a, b = zsc(Xtr, Xte)
        yp[te] = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(a, y[~te]).predict(b)
    return bacc(y, yp)


def ident_after_projection(X, grp, n_dir, mode="inlp", seed=0):
    """Identity decodability after removing dirs estimated on the same data
    (an optimistic-for-us check that identity really is gone)."""
    if n_dir == 0:
        return ident_acc(X, grp)
    rng = np.random.default_rng(seed)
    if mode == "inlp":
        dirs = inlp_directions(X, grp, n_dir, seed)
    else:
        A = rng.standard_normal((n_dir, X.shape[1]))
        dirs = np.linalg.qr(A.T)[0].T
    return ident_acc(project_out(X, dirs), grp)


# ------------------------------------------------------------------ data
def load_all():
    import cross_species as cs
    ds = {}
    rows = cs.load_all()
    E = np.load(os.path.join(D, "out_cross", "emb_wavlm.npy")).astype(np.float32)
    sp = np.array([r["species"] for r in rows])
    y = np.array([r["pol"] for r in rows]); ind = np.array([r["indiv"] for r in rows])
    eg_c = np.load(os.path.join(D, "out_audit", "egemaps_cats.npy")).astype(np.float32)
    eg_d = np.load(os.path.join(D, "out_audit", "egemaps_dogs.npy")).astype(np.float32)
    c, d = sp == "cat", sp == "dog"
    # cats: the audit's eGeMAPS is over ALL 440 meows; our cross-species subset is B/I only
    ds["cats"] = {"emb": E[9][c], "y": y[c], "grp": ind[c],
                  "eg": eg_c if eg_c.shape[0] == c.sum() else None}
    ds["dogs"] = {"emb": E[9][d], "y": y[d], "grp": ind[d],
                  "eg": eg_d if eg_d.shape[0] == d.sum() else None}
    Ep = np.load(os.path.join(D, "out_pigs", "emb_wavlm.npy")).astype(np.float32)
    mp = json.load(open(os.path.join(D, "out_pigs", "meta.json")))
    eg_p = np.load(os.path.join(D, "out_audit", "egemaps_pigs.npy")).astype(np.float32)
    ds["pigs"] = {"emb": Ep[0], "y": np.array([m["pol"] for m in mp]),
                  "grp": np.array([m["team"] for m in mp]),
                  "eg": eg_p if eg_p.shape[0] == Ep.shape[1] else None}
    return ds


def main():
    ds = load_all()
    res = {}

    print("=" * 74)
    print("1a. FUSION  (held-out group; do eGeMAPS and WavLM make different errors?)")
    print("=" * 74)
    for name, d in ds.items():
        emb, y, grp, eg = d["emb"], d["y"], d["grp"], d["eg"]
        r = {"n": int(len(y)), "wavlm": ctx_loGo(emb, y, grp)}
        line = f"  {name:<6s} n={len(y):<5d} wavlm {r['wavlm']:.3f}"
        if eg is not None and eg.shape[0] == len(y):
            eg = np.nan_to_num(eg)
            r["egemaps"] = ctx_loGo(eg, y, grp)
            zc = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-8)
            r["fused"] = ctx_loGo(np.hstack([zc(emb), zc(eg)]), y, grp)
            best = max(r["wavlm"], r["egemaps"])
            r["fusion_gain"] = r["fused"] - best
            line += f"   egemaps {r['egemaps']:.3f}   FUSED {r['fused']:.3f}   gain {r['fused']-best:+.3f}"
        else:
            line += "   (eGeMAPS row-count mismatch, skipped)"
        print(line); res[f"fusion_{name}"] = r

    print()
    print("=" * 74)
    print("1b. IDENTITY-SUBSPACE REMOVAL (INLP) — both numbers, plus a random control")
    print("=" * 74)
    for name, d in ds.items():
        emb, y, grp = d["emb"], d["y"], d["grp"]
        print(f"\n  {name}  ({len(np.unique(grp))} groups, chance ctx 0.500, "
              f"chance id {1/len(np.unique(grp)):.3f})")
        print(f"  {'rank':>5} | {'ctx (INLP)':>10} {'id (INLP)':>10} | {'ctx (rand)':>10} {'id (rand)':>10}")
        rows = []
        for k in [0, 1, 2, 4, 8, 16, 32]:
            ci = loGo_with_projection(emb, y, grp, k, "inlp")
            ii = ident_after_projection(emb, grp, k, "inlp")
            cr = loGo_with_projection(emb, y, grp, k, "rand") if k else ci
            ir = ident_after_projection(emb, grp, k, "rand") if k else ii
            rows.append({"rank": k, "ctx_inlp": ci, "id_inlp": ii, "ctx_rand": cr, "id_rand": ir})
            print(f"  {k:>5} | {ci:>10.3f} {ii:>10.3f} | {cr:>10.3f} {ir:>10.3f}")
        res[f"inlp_{name}"] = rows
        base = rows[0]
        best = max(rows[1:], key=lambda r: r["ctx_inlp"])
        drop_id = base["id_inlp"] - best["id_inlp"]
        gain_ctx = best["ctx_inlp"] - base["ctx_inlp"]
        print(f"   -> best rank {best['rank']}: context {gain_ctx:+.3f}, identity {-drop_id:+.3f}")
        if gain_ctx > 0.01 and drop_id > 0.05:
            print("      WIN: identity removed and context improved")
        elif abs(gain_ctx) <= 0.01 and drop_id > 0.05:
            print("      PARTIAL: identity removed, context held (useful, not a gain)")
        else:
            print("      NO: context did not survive / identity did not fall")

    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print(f"\nwrote {OUT}/results.json")


if __name__ == "__main__":
    main()
