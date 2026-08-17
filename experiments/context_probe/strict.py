"""
Two cheap controls that remove two of the four stated dangers.

DANGER 3 - inverted class imbalance.
  Cats are negative-heavy (221/127), dogs negative-light (99/209). That is
  exactly the configuration where a miscalibrated probe can look like it
  transfers. Fix: subsample BOTH species to equal class sizes, repeat 200x
  with different subsamples, report the distribution.

DANGER 4 - transductive z-scoring.
  The target species' mean/SD currently use the test data. Standard practice
  in domain adaptation, but it should not be load-bearing. Fix: strict
  leave-one-target-animal-out scaling -- for each held-out animal, compute the
  target scaler from the OTHER target animals only, and never touch the
  held-out animal's statistics.

Both run on cached WavLM embeddings. ~2 minutes.
"""
from __future__ import annotations

import json, os, warnings
import numpy as np

warnings.filterwarnings("ignore")
D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "out_cross")
N_REP = 200
N_PERM = 200


def fit_predict(Xa, ya, Xb):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(Xa, ya).predict(Xb)


def bacc(a, b):
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(a, b)


def zfit(X):
    return X.mean(0), X.std(0) + 1e-8


def main():
    import cross_species as cs
    rows = cs.load_all()
    sp = np.array([r["species"] for r in rows])
    y = np.array([r["pol"] for r in rows])
    ind = np.array([r["indiv"] for r in rows])
    E = np.load(os.path.join(OUT, "emb_wavlm.npy")).astype(np.float32)
    c, d = sp == "cat", sp == "dog"
    rng = np.random.default_rng(11)
    out = {}

    # ------------------------------------------------ DANGER 3: equal class sizes
    print("=== equal class sizes, 200 resamples (dog -> cat, mean over all layers) ===")
    ic, yc = np.where(c)[0], y[c]
    idd, yd = np.where(d)[0], y[d]
    n_c = min((yc == 0).sum(), (yc == 1).sum())
    n_d = min((yd == 0).sum(), (yd == 1).sum())
    print(f"  subsampling to {n_c}/class in cats, {n_d}/class in dogs")

    obs = []
    for _ in range(N_REP):
        pc = np.concatenate([rng.choice(ic[yc == k], n_c, replace=False) for k in (0, 1)])
        pd = np.concatenate([rng.choice(idd[yd == k], n_d, replace=False) for k in (0, 1)])
        acc = []
        for li in range(E.shape[0]):
            Xc, Xd = E[li][pc], E[li][pd]
            mc, sc_ = zfit(Xc); md, sd_ = zfit(Xd)
            acc.append(bacc(y[pc], fit_predict((Xd - md) / sd_, y[pd], (Xc - mc) / sc_)))
        obs.append(np.mean(acc))
    obs = np.array(obs)

    null = []
    for _ in range(N_PERM):
        pc = np.concatenate([rng.choice(ic[yc == k], n_c, replace=False) for k in (0, 1)])
        pd = np.concatenate([rng.choice(idd[yd == k], n_d, replace=False) for k in (0, 1)])
        yperm = rng.permutation(y[pd])
        acc = []
        for li in range(E.shape[0]):
            Xc, Xd = E[li][pc], E[li][pd]
            mc, sc_ = zfit(Xc); md, sd_ = zfit(Xd)
            acc.append(bacc(y[pc], fit_predict((Xd - md) / sd_, yperm, (Xc - mc) / sc_)))
        null.append(np.mean(acc))
    null = np.array(null)
    p = float((null >= obs.mean()).mean())
    print(f"  observed {obs.mean():.3f} (sd {obs.std():.3f}, 95% range "
          f"[{np.percentile(obs,2.5):.3f}, {np.percentile(obs,97.5):.3f}])")
    print(f"  null     {null.mean():.3f} (sd {null.std():.3f})   p = {p:.4f}")
    print(f"  [unbalanced original was 0.583, p=0.0033]")
    out["balanced"] = {"obs_mean": float(obs.mean()), "obs_sd": float(obs.std()),
                       "lo": float(np.percentile(obs, 2.5)), "hi": float(np.percentile(obs, 97.5)),
                       "null_mean": float(null.mean()), "p": p, "n_per_class": [int(n_c), int(n_d)]}

    # ------------------------------------------------ DANGER 4: strict scaling
    print("\n=== strict leave-one-target-animal-out scaling (dog -> cat) ===")
    md, sd_ = zfit(E[0][d])  # placeholder, recomputed per layer below
    per_layer = []
    for li in range(E.shape[0]):
        Xd = E[li][d]
        mdl, sdl = zfit(Xd)
        Zd = (Xd - mdl) / sdl
        clf_pred = np.empty(c.sum(), dtype=y.dtype)
        Xc_all, yc_all, ic_all = E[li][c], y[c], ind[c]
        for g in np.unique(ic_all):
            te = ic_all == g
            if (~te).sum() < 10:
                clf_pred[te] = 0; continue
            # scaler from the OTHER cats only -- held-out animal contributes nothing
            m, s = zfit(Xc_all[~te])
            clf_pred[te] = fit_predict(Zd, y[d], (Xc_all[te] - m) / s)
        per_layer.append(bacc(yc_all, clf_pred))
    strict_mean = float(np.mean(per_layer))

    nullstrict = []
    for _ in range(N_PERM):
        yp = rng.permutation(y[d])
        acc = []
        for li in range(E.shape[0]):
            Xd = E[li][d]; mdl, sdl = zfit(Xd); Zd = (Xd - mdl) / sdl
            Xc_all, yc_all, ic_all = E[li][c], y[c], ind[c]
            pr = np.empty(c.sum(), dtype=y.dtype)
            for g in np.unique(ic_all):
                te = ic_all == g
                if (~te).sum() < 10:
                    pr[te] = 0; continue
                m, s = zfit(Xc_all[~te])
                pr[te] = fit_predict(Zd, yp, (Xc_all[te] - m) / s)
            acc.append(bacc(yc_all, pr))
        nullstrict.append(np.mean(acc))
    nullstrict = np.array(nullstrict)
    ps = float((nullstrict >= strict_mean).mean())
    print(f"  strict mean over layers {strict_mean:.3f}   null {nullstrict.mean():.3f}"
          f"±{nullstrict.std():.3f}   p = {ps:.4f}")
    print(f"  per-layer: {' '.join(f'{v:.2f}' for v in per_layer)}")
    print(f"  [transductive original was 0.583, p=0.0033]")
    out["strict_scaling"] = {"mean": strict_mean, "p": ps, "per_layer": [float(v) for v in per_layer],
                             "null_mean": float(nullstrict.mean())}

    json.dump(out, open(os.path.join(OUT, "strict.json"), "w"), indent=1)
    print("\n=== VERDICT ===")
    ok1 = out["balanced"]["p"] < 0.05
    ok2 = out["strict_scaling"]["p"] < 0.05
    print(f"  equal class sizes : {'SURVIVES' if ok1 else 'DOES NOT SURVIVE'} (p={out['balanced']['p']:.4f})")
    print(f"  strict scaling    : {'SURVIVES' if ok2 else 'DOES NOT SURVIVE'} (p={ps:.4f})")
    if ok1 and ok2:
        print("  -> dangers 3 and 4 are eliminated. Remaining: two-corpus channel, one contrast each.")


if __name__ == "__main__":
    main()
