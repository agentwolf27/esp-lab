"""
Does the dog->cat affect transfer replicate across encoders, and is it carried
by more than a couple of animals?

Three tests, each aimed at a way the layer-9 WavLM result could still be luck:

 A. REPLICATION. Re-run the whole thing with HuBERT and wav2vec2. A finding
    that only exists in one model is a property of that model, not of animals.

 B. SELECTION-FREE STATISTIC. Instead of reporting the best layer, use the
    mean transfer accuracy over ALL layers, and permute to get its null.
    Nothing is chosen after seeing the answer.

 C. BOOTSTRAP OVER TARGET INDIVIDUALS. Resample the target species' animals
    with replacement (whole animals, not clips) and recompute. If the 95% CI
    includes 0.5, the effect rests on a few individuals.
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "out_cross")
N_PERM = 300
N_BOOT = 1000
MODELS = {"wavlm": "microsoft/wavlm-base-plus",
          "hubert": "facebook/hubert-base-ls960",
          "wav2vec2": "facebook/wav2vec2-base"}


def fit_predict(Xa, ya, Xb):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(Xa, ya).predict(Xb)


def zscore_within(X, sp):
    Z = X.copy()
    for s in np.unique(sp):
        m = sp == s
        Z[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-8)
    return Z


def bacc(yt, yp):
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(yt, yp)


def main():
    import cross_species as cs

    rows = cs.load_all()
    sp = np.array([r["species"] for r in rows])
    y = np.array([r["pol"] for r in rows])
    ind = np.array([r["indiv"] for r in rows])
    c, d = sp == "cat", sp == "dog"
    rng = np.random.default_rng(7)
    report = {}

    for name, mid in MODELS.items():
        cache = os.path.join(OUT, f"emb_{name}.npy")
        if os.path.exists(cache):
            E = np.load(cache).astype(np.float32)
        else:
            print(f"embedding with {name}...")
            E = cs.embed(rows, mid)
            np.save(cache, E.astype(np.float16))
        L = E.shape[0]
        print(f"\n=== {name}  ({L} layers) ===")

        # --- B: selection-free mean over all layers, with permutation null
        obs_cd, obs_dc = [], []
        for li in range(L):
            Z = zscore_within(E[li], sp)
            obs_cd.append(bacc(y[d], fit_predict(Z[c], y[c], Z[d])))
            obs_dc.append(bacc(y[c], fit_predict(Z[d], y[d], Z[c])))
        m_cd, m_dc = float(np.mean(obs_cd)), float(np.mean(obs_dc))

        null_cd, null_dc = [], []
        for _ in range(N_PERM):
            pc, pd = rng.permutation(y[c]), rng.permutation(y[d])
            a = b = 0.0
            for li in range(L):
                Z = zscore_within(E[li], sp)
                a += bacc(y[d], fit_predict(Z[c], pc, Z[d]))
                b += bacc(y[c], fit_predict(Z[d], pd, Z[c]))
            null_cd.append(a / L); null_dc.append(b / L)
        null_cd, null_dc = np.array(null_cd), np.array(null_dc)
        p_cd = float((null_cd >= m_cd).mean()); p_dc = float((null_dc >= m_dc).mean())
        print(f"  mean over all layers   cat->dog {m_cd:.3f} (null {null_cd.mean():.3f}"
              f"±{null_cd.std():.3f}, p={p_cd:.4f})")
        print(f"                         dog->cat {m_dc:.3f} (null {null_dc.mean():.3f}"
              f"±{null_dc.std():.3f}, p={p_dc:.4f})")

        # --- C: bootstrap over target individuals, at the best dog->cat layer
        li_best = int(np.argmax(obs_dc))
        Z = zscore_within(E[li_best], sp)
        pred_c = fit_predict(Z[d], y[d], Z[c])          # dog-trained, cat targets
        cat_ids = np.unique(ind[c]); yc = y[c]; ic = ind[c]
        boots = []
        for _ in range(N_BOOT):
            pick = rng.choice(cat_ids, len(cat_ids), replace=True)
            idx = np.concatenate([np.where(ic == g)[0] for g in pick])
            if len(np.unique(yc[idx])) < 2:
                continue
            boots.append(bacc(yc[idx], pred_c[idx]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  dog->cat best layer {li_best}: {obs_dc[li_best]:.3f}  "
              f"bootstrap 95% CI over cats [{lo:.3f}, {hi:.3f}]")

        per_cat = {}
        for g in cat_ids:
            m = ic == g
            if m.sum() >= 5:
                per_cat[str(g)] = float((pred_c[m] == yc[m]).mean())
        above = sum(v > .5 for v in per_cat.values())
        print(f"  per-cat: {above}/{len(per_cat)} above 0.5  "
              f"(median {np.median(list(per_cat.values())):.3f})")

        report[name] = {"layers": L, "mean_cat_to_dog": m_cd, "mean_dog_to_cat": m_dc,
                        "p_cat_to_dog": p_cd, "p_dog_to_cat": p_dc,
                        "null_cd_mean": float(null_cd.mean()), "null_dc_mean": float(null_dc.mean()),
                        "best_dc_layer": li_best, "best_dc": float(obs_dc[li_best]),
                        "boot_ci": [float(lo), float(hi)], "per_cat": per_cat,
                        "per_layer_cd": [float(v) for v in obs_cd],
                        "per_layer_dc": [float(v) for v in obs_dc]}
        json.dump(report, open(os.path.join(OUT, "replication.json"), "w"), indent=1)

    print("\n=== VERDICT ===")
    for k, v in report.items():
        ok_dc = "SURVIVES" if v["p_dog_to_cat"] < 0.05 else "no"
        ok_cd = "SURVIVES" if v["p_cat_to_dog"] < 0.05 else "no"
        print(f"  {k:<9s} dog->cat {v['mean_dog_to_cat']:.3f} p={v['p_dog_to_cat']:.4f} [{ok_dc}]   "
              f"cat->dog {v['mean_cat_to_dog']:.3f} p={v['p_cat_to_dog']:.4f} [{ok_cd}]")


if __name__ == "__main__":
    main()
