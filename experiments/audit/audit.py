"""
Multi-dataset audit of the "encoders hear WHO/WHERE, not WHAT-SITUATION" finding.

For every (dataset, feature-set, layer) we record three numbers:
    heldout   situation accuracy, leave-one-GROUP-out          (the honest one)
    random    situation accuracy, random stratified 5-fold     (the leaky one)
    identity  group-identity accuracy, random stratified 5-fold

    inflation = random - heldout

Group = cat (21) / dog (10) / recording lab (6) / bat emitter (10).
All situation scores are BALANCED accuracy (chance = 1/n_classes).
Identity is plain accuracy (chance = 1/n_groups; the majority-group rate is
also recorded, since group sizes are unequal).

Estimator conventions are inherited from the scripts that produced the
already-published numbers, so the audit table reproduces them exactly:
    "cat style"  per-fold StandardScaler + LogisticRegression(C=1, max_iter=3000)
                 (ctx/encoders.py -- used for cats, dogs, bats)
    "pig style"  global z-score + LogisticRegression(C=1, class_weight=balanced)
                 (ctx/pigs_run.py -- used for pigs; LOTO is scored only on the
                 four labs that recorded both valences)
Dogs are additionally run in pig style as a sensitivity check.

    python audit.py points|law|recovery|report|all
"""
from __future__ import annotations

import json, os, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
CTX = os.path.dirname(HERE)
SEED = 0


# ----------------------------------------------------------------- estimators
def _lr(balanced=False):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=3000, C=1.0,
                              class_weight="balanced" if balanced else None)


def _zglobal(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


def heldout_group(X, y, grp, style):
    """Leave-one-group-out. Returns (balanced acc, per-group acc dict).

    pig style skips groups that do not contain both classes (they cannot be
    scored with balanced accuracy) -- this is the published convention.
    """
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    yp = np.full(len(y), None, dtype=object)
    for g in np.unique(grp):
        te = grp == g
        if style == "pig" and len(np.unique(y[te])) < 2:
            continue
        if len(np.unique(y[~te])) < 2:
            continue
        if style == "pig":
            Z = _zglobal(X)
            yp[te] = _lr(True).fit(Z[~te], y[~te]).predict(Z[te])
        else:
            sc = StandardScaler().fit(X[~te])
            yp[te] = _lr(False).fit(sc.transform(X[~te]), y[~te]).predict(sc.transform(X[te]))
    m = np.array([v is not None for v in yp])
    per = {str(g): float(balanced_accuracy_score(y[(grp == g) & m], yp[(grp == g) & m].astype(y.dtype)))
           for g in np.unique(grp[m])}
    return float(balanced_accuracy_score(y[m], yp[m].astype(y.dtype))), per, int(m.sum())


def random_kfold(X, y, style, k=5, balanced_metric=True):
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    yp = np.empty_like(y)
    Z = _zglobal(X) if style == "pig" else None
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=SEED).split(X, y):
        if style == "pig":
            yp[te] = _lr(True).fit(Z[tr], y[tr]).predict(Z[te])
        else:
            sc = StandardScaler().fit(X[tr])
            yp[te] = _lr(False).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    if balanced_metric:
        return float(balanced_accuracy_score(y, yp))
    return float((yp == y).mean())


def identity_acc(X, grp, style, min_n=5):
    """Group identity from the same features, random stratified 5-fold.
    Returns (plain acc, balanced acc, n groups, n clips used)."""
    from sklearn.metrics import balanced_accuracy_score
    keep = np.isin(grp, [g for g in np.unique(grp) if (grp == g).sum() >= min_n])
    Xi, gi = X[keep], grp[keep]
    yp = random_kfold_pred(Xi, gi, style)
    return (float((yp == gi).mean()), float(balanced_accuracy_score(gi, yp)),
            int(len(np.unique(gi))), int(keep.sum()))


def random_kfold_pred(X, y, style, k=5):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    yp = np.empty_like(y)
    Z = _zglobal(X) if style == "pig" else None
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=SEED).split(X, y):
        if style == "pig":
            yp[te] = _lr(True).fit(Z[tr], y[tr]).predict(Z[te])
        else:
            sc = StandardScaler().fit(X[tr])
            yp[te] = _lr(False).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    return yp


def three_numbers(X, y, grp, style):
    ho, per, n_scored = heldout_group(X, y, grp, style)
    rd = random_kfold(X, y, style)
    idp, idb, ng, nid = identity_acc(X, grp, style)
    return {"heldout": ho, "random": rd, "inflation": rd - ho,
            "identity": idp, "identity_bal": idb,
            "n_groups_id": ng, "n_clips_id": nid, "n_scored_heldout": n_scored,
            "per_group": per}


# ----------------------------------------------------------------- data
def load_cats():
    meta = json.load(open(os.path.join(CTX, "out_enc", "meta.json")))["meta"]
    y = np.array([m["ctx"] for m in meta])
    grp = np.array([m["cat"] for m in meta])
    return y, grp


def load_dogs():
    meta = json.load(open(os.path.join(HERE, "meta_dogs.json")))
    y = np.array([m["ctx"] for m in meta])
    grp = np.array([m["dog"] for m in meta])
    return y, grp


def load_pigs():
    meta = json.load(open(os.path.join(CTX, "out_pigs", "meta.json")))
    y = np.array([m["pol"] for m in meta])
    grp = np.array([m["team"] for m in meta])
    A = np.array([m["acoustic"] for m in meta], dtype=np.float32)
    A = np.where(np.isnan(A), np.nanmean(A, 0), A)
    return y, grp, A


def load_bats():
    """Exploratory: the 453 BEANS-bats clips in our cached 2,000-clip subset
    that carry a behavioural-context annotation.  Files with more than one
    annotation row are dropped as ambiguous."""
    import pandas as pd
    sub = pd.read_csv(os.path.join(CTX, "bats", "subset2000.csv"))
    j = pd.read_csv(os.path.join(CTX, "bats", "bats_joined.csv"))
    j = j.drop_duplicates(subset=["File Name"], keep=False)          # ambiguous -> out
    m = sub.merge(j[["File Name", "ctx_name"]], on="File Name", how="left")
    assert len(m) == len(sub)
    keep_ctx = ["sleeping", "fighting", "isolation", "feeding"]
    isin = np.asarray(m["ctx_name"].isin(keep_ctx).values, dtype=bool)
    idx = np.where(isin)[0]
    ctx = np.array([str(v) for v in m["ctx_name"].to_numpy(dtype=object)[idx]])
    emt = np.array([str(v) for v in sub["Emitter"].to_numpy(dtype=object)[idx]])
    return idx, ctx, emt


def emb(path, layer):
    E = np.load(path, mmap_mode="r")
    return np.asarray(E[layer], dtype=np.float32)


def n_layers(path):
    return int(np.load(path, mmap_mode="r").shape[0])


# ----------------------------------------------------------------- step 1
def step_points():
    """Every (dataset, feature-set, layer) point.  Cats and pigs WavLM-family
    numbers are re-used from the published runs (out_enc / out_pigs) and
    spot-verified here; everything else is computed fresh."""
    pts = []
    log = []

    # ---- cats: reuse out_enc/results.json (mfcc, wavlm, hubert, wav2vec2, aves-bio)
    enc = json.load(open(os.path.join(CTX, "out_enc", "results.json")))
    for name, v in enc.items():
        if name == "egemaps":
            continue                                    # recomputed below at C=1.0
        for r in v["layers"]:
            pts.append({"dataset": "cats", "features": name, "layer": r["layer"],
                        "dim": r["dim"], "heldout": r["ctx_loco"], "random": r["ctx_rand"],
                        "inflation": r["ctx_rand"] - r["ctx_loco"], "identity": r["identity"],
                        "source": "out_enc/results.json"})

    # ---- pigs: reuse out_pigs/results.json (wavlm) + paper's 18 features
    pg = json.load(open(os.path.join(CTX, "out_pigs", "results.json")))
    for r in pg["within"]:
        pts.append({"dataset": "pigs", "features": "wavlm", "layer": r["layer"], "dim": 768,
                    "heldout": r["loto"], "random": r["random"],
                    "inflation": r["random"] - r["loto"], "identity": r["team_id"],
                    "source": "out_pigs/results.json"})

    y_p, g_p, A = load_pigs()
    t0 = time.time()
    print("verifying pig WavLM layer 0 reproduces the published numbers...", flush=True)
    v = three_numbers(emb(os.path.join(CTX, "out_pigs", "emb_wavlm.npy"), 0), y_p, g_p, "pig")
    pub = pg["within"][0]
    log.append({"check": "pigs_wavlm_L0", "recomputed": [v["heldout"], v["random"], v["identity"]],
                "published": [pub["loto"], pub["random"], pub["team_id"]]})
    print(f"  recomputed {v['heldout']:.4f}/{v['random']:.4f}/{v['identity']:.4f}  "
          f"published {pub['loto']:.4f}/{pub['random']:.4f}/{pub['team_id']:.4f}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    print("pigs / paper's 18 acoustic features ...", flush=True)
    v = three_numbers(A, y_p, g_p, "pig")
    pts.append({"dataset": "pigs", "features": "acoustic18", "layer": 0, "dim": A.shape[1],
                "heldout": v["heldout"], "random": v["random"], "inflation": v["inflation"],
                "identity": v["identity"], "source": "computed"})
    print(f"  {v['heldout']:.3f} / {v['random']:.3f} / id {v['identity']:.3f}", flush=True)

    # ---- pigs eGeMAPS
    p = os.path.join(HERE, "egemaps_pigs.npy")
    if os.path.exists(p):
        print("pigs / eGeMAPS ...", flush=True)
        X = np.load(p)
        v = three_numbers(X, y_p, g_p, "pig")
        pts.append({"dataset": "pigs", "features": "egemaps", "layer": 0, "dim": X.shape[1],
                    "heldout": v["heldout"], "random": v["random"], "inflation": v["inflation"],
                    "identity": v["identity"], "source": "computed"})
        print(f"  {v['heldout']:.3f} / {v['random']:.3f} / id {v['identity']:.3f}", flush=True)

    # ---- cats eGeMAPS (recomputed with the encoder-protocol C=1.0)
    p = os.path.join(HERE, "egemaps_cats.npy")
    y_c, g_c = load_cats()
    if os.path.exists(p):
        print("cats / eGeMAPS ...", flush=True)
        X = np.load(p)
        assert len(X) == len(y_c)
        v = three_numbers(X, y_c, g_c, "cat")
        pts.append({"dataset": "cats", "features": "egemaps", "layer": 0, "dim": X.shape[1],
                    "heldout": v["heldout"], "random": v["random"], "inflation": v["inflation"],
                    "identity": v["identity"], "source": "computed"})
        print(f"  {v['heldout']:.3f} / {v['random']:.3f} / id {v['identity']:.3f}", flush=True)

    # ---- dogs WavLM, all layers
    y_d, g_d = load_dogs()
    pdog = os.path.join(HERE, "emb_dogs_wavlm.npy")
    L = n_layers(pdog)
    print(f"dogs / WavLM, {L} layers  (n={len(y_d)}, {len(set(g_d))} dogs)", flush=True)
    for li in range(L):
        X = emb(pdog, li)
        v = three_numbers(X, y_d, g_d, "cat")
        pts.append({"dataset": "dogs", "features": "wavlm", "layer": li, "dim": X.shape[1],
                    "heldout": v["heldout"], "random": v["random"], "inflation": v["inflation"],
                    "identity": v["identity"], "identity_bal": v["identity_bal"],
                    "per_group": v["per_group"], "source": "computed"})
        print(f"  L{li:<2d} heldout {v['heldout']:.3f}  random {v['random']:.3f}  "
              f"identity {v['identity']:.3f}", flush=True)
        # sensitivity: pig-style estimator (class-weighted) on the same layer
        vb = three_numbers(X, y_d, g_d, "pig")
        pts.append({"dataset": "dogs", "features": "wavlm", "layer": li, "dim": X.shape[1],
                    "heldout": vb["heldout"], "random": vb["random"], "inflation": vb["inflation"],
                    "identity": vb["identity"], "source": "computed", "variant": "balanced"})

    # ---- dogs eGeMAPS
    p = os.path.join(HERE, "egemaps_dogs.npy")
    if os.path.exists(p):
        print("dogs / eGeMAPS ...", flush=True)
        X = np.load(p)
        v = three_numbers(X, y_d, g_d, "cat")
        pts.append({"dataset": "dogs", "features": "egemaps", "layer": 0, "dim": X.shape[1],
                    "heldout": v["heldout"], "random": v["random"], "inflation": v["inflation"],
                    "identity": v["identity"], "identity_bal": v["identity_bal"],
                    "per_group": v["per_group"], "source": "computed"})
        print(f"  {v['heldout']:.3f} / {v['random']:.3f} / id {v['identity']:.3f}", flush=True)
        vb = three_numbers(X, y_d, g_d, "pig")
        pts.append({"dataset": "dogs", "features": "egemaps", "layer": 0, "dim": X.shape[1],
                    "heldout": vb["heldout"], "random": vb["random"], "inflation": vb["inflation"],
                    "identity": vb["identity"], "source": "computed", "variant": "balanced"})

    # ---- dogs MFCC-85 (same handcrafted floor used on cats)
    print("dogs / MFCC-85 ...", flush=True)
    sys.path.insert(0, CTX)
    import soundfile as sf
    from scipy.signal import resample_poly
    from context_probe import clip_vector
    meta = json.load(open(os.path.join(HERE, "meta_dogs.json")))
    Xm = []
    for m in meta:
        x, sr = sf.read(os.path.join(CTX, "dogs", "audio", m["file"]), dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        if sr != 16000:
            g = np.gcd(int(sr), 16000)
            x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
        Xm.append(clip_vector(x[: 6 * 16000]))
    Xm = np.nan_to_num(np.stack(Xm).astype(np.float32))
    np.save(os.path.join(HERE, "mfcc_dogs.npy"), Xm)
    v = three_numbers(Xm, y_d, g_d, "cat")
    pts.append({"dataset": "dogs", "features": "mfcc", "layer": 0, "dim": Xm.shape[1],
                "heldout": v["heldout"], "random": v["random"], "inflation": v["inflation"],
                "identity": v["identity"], "source": "computed"})
    print(f"  {v['heldout']:.3f} / {v['random']:.3f} / id {v['identity']:.3f}", flush=True)

    # ---- bats (exploratory)
    try:
        idx, y_b, g_b = load_bats()
        print(f"bats / WavLM (exploratory, n={len(idx)}, {len(set(g_b))} emitters, "
              f"{len(set(y_b))} contexts)", flush=True)
        pb = os.path.join(CTX, "out_bats", "emb_wavlm_native.npy")
        for li in range(n_layers(pb)):
            X = emb(pb, li)[idx]
            v = three_numbers(X, y_b, g_b, "cat")
            pts.append({"dataset": "bats", "features": "wavlm", "layer": li, "dim": X.shape[1],
                        "heldout": v["heldout"], "random": v["random"], "inflation": v["inflation"],
                        "identity": v["identity"], "source": "computed", "exploratory": True})
            print(f"  L{li:<2d} heldout {v['heldout']:.3f}  random {v['random']:.3f}  "
                  f"identity {v['identity']:.3f}", flush=True)
    except Exception as e:                                        # noqa: BLE001
        print("bats skipped:", type(e).__name__, e, flush=True)

    json.dump({"points": pts, "checks": log},
              open(os.path.join(HERE, "points.json"), "w"), indent=1)
    print(f"\nwrote points.json  ({len(pts)} points)")


# ----------------------------------------------------------------- step 1b
def step_extras():
    """Appended afterwards: bats (exploratory) + a duration-only floor for dogs."""
    P = json.load(open(os.path.join(HERE, "points.json")))
    pts = [p for p in P["points"] if p["dataset"] != "bats" and p["features"] != "duration"]

    # dogs, duration only (the trivial confound: are the contexts just different
    # recording lengths?)  Same floor test pigs_run.py ran on pigs.
    meta = json.load(open(os.path.join(HERE, "meta_dogs.json")))
    y_d, g_d = load_dogs()
    Xd = np.log(np.array([[m["dur_full"]] for m in meta], dtype=np.float32) + 1e-3)
    v = three_numbers(Xd, y_d, g_d, "cat")
    pts.append({"dataset": "dogs", "features": "duration", "layer": 0, "dim": 1,
                "heldout": v["heldout"], "random": v["random"], "inflation": v["inflation"],
                "identity": v["identity"], "source": "computed", "floor": True})
    print(f"dogs / log-duration only: heldout {v['heldout']:.3f}  random {v['random']:.3f}  "
          f"identity {v['identity']:.3f}", flush=True)

    # bats, exploratory
    idx, y_b, g_b = load_bats()
    import collections
    print(f"bats: n={len(idx)}  contexts {dict(collections.Counter(y_b))}  "
          f"emitters {len(set(g_b))}", flush=True)
    pb = os.path.join(CTX, "out_bats", "emb_wavlm_native.npy")
    for li in range(n_layers(pb)):
        X = emb(pb, li)[idx]
        v = three_numbers(X, y_b, g_b, "cat")
        pts.append({"dataset": "bats", "features": "wavlm", "layer": li, "dim": X.shape[1],
                    "heldout": v["heldout"], "random": v["random"], "inflation": v["inflation"],
                    "identity": v["identity"], "source": "computed", "exploratory": True})
        print(f"  L{li:<2d} heldout {v['heldout']:.3f}  random {v['random']:.3f}  "
              f"identity {v['identity']:.3f}", flush=True)

    P["points"] = pts
    json.dump(P, open(os.path.join(HERE, "points.json"), "w"), indent=1)
    print(f"wrote points.json ({len(pts)} points)")


# ----------------------------------------------------------------- step 2
def primary_points(P, include_bats=False):
    out = []
    for p in P["points"]:
        if p.get("variant"):                      # sensitivity re-runs
            continue
        if p.get("floor"):                        # duration-only floor
            continue
        if p["dataset"] == "bats" and not include_bats:
            continue
        out.append(p)
    return out


def corr_block(x, y):
    from scipy import stats
    pr = stats.pearsonr(x, y)
    sp = stats.spearmanr(x, y)
    return {"n": int(len(x)), "pearson_r": float(pr[0]), "pearson_p": float(pr[1]),
            "spearman_rho": float(sp[0]), "spearman_p": float(sp[1])}


def step_law():
    from scipy import stats
    P = json.load(open(os.path.join(HERE, "points.json")))
    res = {}

    for tag, inc in [("primary_cats_dogs_pigs", False), ("with_bats", True)]:
        pts = primary_points(P, include_bats=inc)
        ds = np.array([p["dataset"] for p in pts])
        idn = np.array([p["identity"] for p in pts], dtype=float)
        gap = np.array([p["inflation"] for p in pts], dtype=float)
        ho = np.array([p["heldout"] for p in pts], dtype=float)
        rd = np.array([p["random"] for p in pts], dtype=float)
        b = {"pooled": corr_block(idn, gap),
             "identity_vs_random": corr_block(idn, rd),
             "identity_vs_heldout": corr_block(idn, ho)}

        # within-dataset partial correlation (remove each dataset's mean from
        # both variables).  df = n - n_datasets - 1.
        xi, gi = idn.copy(), gap.copy()
        for d in np.unique(ds):
            m = ds == d
            xi[m] -= xi[m].mean(); gi[m] -= gi[m].mean()
        r = float(np.corrcoef(xi, gi)[0, 1])
        n, k = len(xi), len(np.unique(ds))
        dfree = n - k - 1
        t = r * np.sqrt(dfree / max(1e-12, 1 - r ** 2))
        b["within_dataset_partial"] = {"n": n, "df": int(dfree), "r": r,
                                       "p": float(2 * stats.t.sf(abs(t), dfree))}
        # per dataset
        b["per_dataset"] = {}
        for d in np.unique(ds):
            m = ds == d
            if m.sum() >= 3:
                b["per_dataset"][d] = corr_block(idn[m], gap[m])
        # cluster level: one point per (dataset, feature-set), layers averaged
        keys = sorted({(p["dataset"], p["features"]) for p in pts})
        cx = np.array([np.mean([p["identity"] for p in pts
                                if (p["dataset"], p["features"]) == kk]) for kk in keys])
        cy = np.array([np.mean([p["inflation"] for p in pts
                                if (p["dataset"], p["features"]) == kk]) for kk in keys])
        b["cluster_level"] = corr_block(cx, cy)
        b["cluster_level"]["keys"] = ["/".join(kk) for kk in keys]
        b["cluster_level"]["identity"] = cx.tolist()
        b["cluster_level"]["inflation"] = cy.tolist()

        # control 1: only the 768-d neural points, so the correlation cannot be
        # explained by "handcrafted 85/88-d vs neural 768-d capacity".
        m768 = np.array([p["dim"] == 768 for p in pts])
        b["neural_only"] = corr_block(idn[m768], gap[m768])
        xi2, gi2 = idn[m768].copy(), gap[m768].copy()
        ds2 = ds[m768]
        for d in np.unique(ds2):
            m = ds2 == d
            xi2[m] -= xi2[m].mean(); gi2[m] -= gi2[m].mean()
        r2 = float(np.corrcoef(xi2, gi2)[0, 1])
        df2 = len(xi2) - len(np.unique(ds2)) - 1
        t2 = r2 * np.sqrt(df2 / max(1e-12, 1 - r2 ** 2))
        b["neural_only_within_dataset"] = {"n": int(len(xi2)), "df": int(df2), "r": r2,
                                           "p": float(2 * stats.t.sf(abs(t2), df2))}

        # control 2: cats only, one point per feature-set (the n=6 analogue of
        # the earlier 5-encoder analysis that gave r=0.81, p=0.094)
        ck = sorted({p["features"] for p in pts if p["dataset"] == "cats"})
        if len(ck) >= 3:
            ax = np.array([np.mean([p["identity"] for p in pts
                                    if p["dataset"] == "cats" and p["features"] == f]) for f in ck])
            ay = np.array([np.mean([p["inflation"] for p in pts
                                    if p["dataset"] == "cats" and p["features"] == f]) for f in ck])
            b["cats_encoder_level"] = corr_block(ax, ay)
            b["cats_encoder_level"]["keys"] = ck
        # control 3: within-ENCODER only.  Centre inside each (dataset,
        # feature-set) cluster, so all that is left is layer-to-layer variation
        # inside one frozen encoder.  This is the finest-grained form of the
        # claim and the one least able to be explained by anything else.
        key_arr = np.array([f"{p['dataset']}/{p['features']}" for p in pts])
        xi3, gi3 = idn.copy(), gap.copy()
        multi = np.zeros(len(pts), dtype=bool)
        for kk in np.unique(key_arr):
            m = key_arr == kk
            if m.sum() < 3:
                continue
            multi |= m
            xi3[m] -= xi3[m].mean(); gi3[m] -= gi3[m].mean()
        r3 = float(np.corrcoef(xi3[multi], gi3[multi])[0, 1])
        df3 = int(multi.sum()) - len(np.unique(key_arr[multi])) - 1
        t3 = r3 * np.sqrt(df3 / max(1e-12, 1 - r3 ** 2))
        b["within_encoder_partial"] = {"n": int(multi.sum()), "df": df3, "r": r3,
                                       "p": float(2 * stats.t.sf(abs(t3), df3)),
                                       "n_clusters": int(len(np.unique(key_arr[multi])))}
        b["per_encoder"] = {}
        for kk in np.unique(key_arr):
            m = key_arr == kk
            if m.sum() >= 5:
                b["per_encoder"][kk] = corr_block(idn[m], gap[m])

        # control 4: leverage.  Drop the single most influential point
        # (largest |x - mean(x)| residual product) and refit.
        d2 = (idn - idn.mean()) ** 2
        drop = int(np.argmax(d2))
        keep = np.ones(len(idn), bool); keep[drop] = False
        b["drop_highest_leverage"] = corr_block(idn[keep], gap[keep])
        b["drop_highest_leverage"]["dropped"] = f"{pts[drop]['dataset']}/{pts[drop]['features']}"
        # and a jackknife over whole feature-set clusters
        jk = []
        for kk in np.unique(key_arr):
            m = key_arr != kk
            jk.append(float(stats.pearsonr(idn[m], gap[m])[0]))
        b["jackknife_drop_one_featureset"] = {"min_r": float(np.min(jk)), "max_r": float(np.max(jk)),
                                              "n_drops": len(jk)}

        # OLS fit for the plot
        sl, ic, rv, pv, se = stats.linregress(idn, gap)
        b["fit"] = {"slope": float(sl), "intercept": float(ic), "r": float(rv),
                    "p": float(pv), "stderr": float(se)}
        res[tag] = b

    json.dump(res, open(os.path.join(HERE, "law.json"), "w"), indent=1)
    for tag, b in res.items():
        print(f"\n=== {tag} ===")
        p = b["pooled"]
        print(f"  pooled            n={p['n']}  Pearson r={p['pearson_r']:.3f} p={p['pearson_p']:.2e}"
              f"   Spearman rho={p['spearman_rho']:.3f} p={p['spearman_p']:.2e}")
        w = b["within_dataset_partial"]
        print(f"  within-dataset    n={w['n']} df={w['df']}  r={w['r']:.3f} p={w['p']:.2e}")
        c = b["cluster_level"]
        print(f"  cluster level     n={c['n']}  r={c['pearson_r']:.3f} p={c['pearson_p']:.3f}"
              f"   rho={c['spearman_rho']:.3f} p={c['spearman_p']:.3f}")
        for d, v in b["per_dataset"].items():
            print(f"    {d:<5s} n={v['n']:<3d} r={v['pearson_r']:.3f} p={v['pearson_p']:.3f}"
                  f"   rho={v['spearman_rho']:.3f} p={v['spearman_p']:.3f}")
        we = b["within_encoder_partial"]
        print(f"  within-encoder    n={we['n']} ({we['n_clusters']} clusters) df={we['df']}  "
              f"r={we['r']:.3f} p={we['p']:.2e}")
        for kk, v in b["per_encoder"].items():
            print(f"    {kk:<18s} n={v['n']:<3d} r={v['pearson_r']:+.3f} p={v['pearson_p']:.3f}")
        dl = b["drop_highest_leverage"]; jk = b["jackknife_drop_one_featureset"]
        print(f"  drop {dl['dropped']:<18s} r={dl['pearson_r']:.3f} p={dl['pearson_p']:.2e}"
              f"   jackknife r range [{jk['min_r']:.3f}, {jk['max_r']:.3f}]")
        nn = b["neural_only"]; nw = b["neural_only_within_dataset"]
        print(f"  neural(768d) only n={nn['n']}  r={nn['pearson_r']:.3f} p={nn['pearson_p']:.2e}"
              f"   within-dataset r={nw['r']:.3f} p={nw['p']:.2e}")
        if "cats_encoder_level" in b:
            ce = b["cats_encoder_level"]
            print(f"  cats, per feature-set  n={ce['n']}  r={ce['pearson_r']:.3f} "
                  f"p={ce['pearson_p']:.3f}")
        print(f"  identity vs random  r={b['identity_vs_random']['pearson_r']:.3f} "
              f"p={b['identity_vs_random']['pearson_p']:.2e}")
        print(f"  identity vs heldout r={b['identity_vs_heldout']['pearson_r']:.3f} "
              f"p={b['identity_vs_heldout']['pearson_p']:.2e}")


# ----------------------------------------------------------------- step 3
def recovery_one(X, y, grp, style, ks, seeds, scored_groups=None):
    """How much does k labelled clips from the held-out group buy?

    Per group: a FIXED stratified test half is set aside once; the other half is
    the adaptation pool.  Every k is evaluated on that same test half, so the
    k=0 baseline and the k>0 models are strictly comparable.  k is capped at the
    pool size (cats simply do not have 50 clips per animal), and the effective k
    actually used is reported.
    """
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler

    groups = [g for g in np.unique(grp)
              if (scored_groups is None or g in scored_groups)]
    # fixed test half per group
    test_idx, pool_idx = {}, {}
    for g in groups:
        gi = np.where(grp == g)[0]
        yg = y[gi]
        if len(np.unique(yg)) < 2 or len(gi) < 4:
            continue
        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
            a, b = next(sss.split(gi.reshape(-1, 1), yg))
        except ValueError:
            # a singleton class in this group -> plain random half split.
            # Harmless: predictions are pooled over groups before scoring.
            perm = np.random.default_rng(SEED).permutation(len(gi))
            h = len(gi) // 2
            a, b = perm[:h], perm[h:]
        pool_idx[g], test_idx[g] = gi[a], gi[b]
    groups = [g for g in groups if g in test_idx]

    Z = _zglobal(X) if style == "pig" else None
    rng = np.random.default_rng(SEED)
    out = {"ks": list(ks), "curve": [], "groups": [str(g) for g in groups],
           "pool_sizes": {str(g): int(len(pool_idx[g])) for g in groups},
           "test_sizes": {str(g): int(len(test_idx[g])) for g in groups}}

    # k = 0 baseline model per group, cached (independent of seed)
    base_pred, base_true = {}, {}
    for g in groups:
        tr = ~np.isin(np.arange(len(y)), np.concatenate([pool_idx[g], test_idx[g]]))
        te = test_idx[g]
        if style == "pig":
            mdl = _lr(True).fit(Z[tr], y[tr]); base_pred[g] = mdl.predict(Z[te])
        else:
            sc = StandardScaler().fit(X[tr])
            base_pred[g] = _lr(False).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
        base_true[g] = y[te]
    base_acc = float(balanced_accuracy_score(np.concatenate([base_true[g] for g in groups]),
                                             np.concatenate([base_pred[g] for g in groups])))

    for k in ks:
        accs, eff = [], []
        n_seeds = 1 if k == 0 else seeds
        for s in range(n_seeds):
            preds, trues = [], []
            for g in groups:
                te = test_idx[g]
                if k == 0:
                    preds.append(base_pred[g]); trues.append(base_true[g]); eff.append(0)
                    continue
                pool = pool_idx[g]
                kk = min(k, len(pool))
                take = rng.choice(pool, size=kk, replace=False)
                eff.append(kk)
                others = np.setdiff1d(np.arange(len(y)),
                                      np.concatenate([pool_idx[g], test_idx[g]]))
                tr = np.concatenate([others, take])
                if style == "pig":
                    preds.append(_lr(True).fit(Z[tr], y[tr]).predict(Z[te]))
                else:
                    sc = StandardScaler().fit(X[tr])
                    preds.append(_lr(False).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te])))
                trues.append(y[te])
            accs.append(float(balanced_accuracy_score(np.concatenate(trues), np.concatenate(preds))))
        out["curve"].append({"k": int(k), "mean": float(np.mean(accs)),
                             "sd": float(np.std(accs)), "n_seeds": n_seeds,
                             "eff_k": float(np.mean(eff)), "accs": accs})
        print(f"    k={k:<4d} eff_k={np.mean(eff):5.1f}  acc {np.mean(accs):.3f} "
              f"+- {np.std(accs):.3f}", flush=True)
    out["base_acc"] = base_acc
    out["random_ceiling"] = random_kfold(X, y, style)
    return out


def step_recovery():
    P = json.load(open(os.path.join(HERE, "points.json")))
    ks = [0, 10, 25, 50, 100]
    res = {}

    def best_layer(dataset, features):
        c = [p for p in P["points"] if p["dataset"] == dataset and p["features"] == features
             and not p.get("variant")]
        return max(c, key=lambda p: p["heldout"])["layer"]

    y_c, g_c = load_cats()
    y_p, g_p, _ = load_pigs()

    bl = best_layer("cats", "wavlm")
    print(f"cats / WavLM L{bl} (best held-out-cat layer)", flush=True)
    res["cats_wavlm"] = recovery_one(emb(os.path.join(CTX, "out_enc", "emb_wavlm.npy"), bl),
                                     y_c, g_c, "cat", ks, seeds=10)
    res["cats_wavlm"]["layer"] = bl

    print("cats / eGeMAPS", flush=True)
    res["cats_egemaps"] = recovery_one(np.load(os.path.join(HERE, "egemaps_cats.npy")),
                                       y_c, g_c, "cat", ks, seeds=10)

    scored = [g for g in np.unique(g_p) if len(np.unique(y_p[g_p == g])) > 1]
    bl = best_layer("pigs", "wavlm")
    print(f"pigs / WavLM L{bl} (best held-out-lab layer); scored labs {scored}", flush=True)
    res["pigs_wavlm"] = recovery_one(emb(os.path.join(CTX, "out_pigs", "emb_wavlm.npy"), bl),
                                     y_p, g_p, "pig", ks, seeds=5, scored_groups=scored)
    res["pigs_wavlm"]["layer"] = bl

    print("pigs / eGeMAPS", flush=True)
    res["pigs_egemaps"] = recovery_one(np.load(os.path.join(HERE, "egemaps_pigs.npy")),
                                       y_p, g_p, "pig", ks, seeds=5, scored_groups=scored)

    json.dump(res, open(os.path.join(HERE, "recovery.json"), "w"), indent=1)
    print("wrote recovery.json")


if __name__ == "__main__":
    {"points": step_points, "extras": step_extras,
     "law": step_law, "recovery": step_recovery}[sys.argv[1]]()
