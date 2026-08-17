"""
Is the cross-species transfer real, or did we fool ourselves?

The raw result was cat->dog 0.606 / dog->cat 0.656 at layer 9. Three ways that
could be an artifact, each tested here:

 1. POST-HOC LAYER SELECTION. We scanned 13 layers and reported the best.
    Fix: nested selection -- choose the layer using ONLY within-species
    cross-validation on the source species, then report transfer at that layer.
    Also report the mean over all layers, which needs no selection at all.

 2. THE NULL IS WIDER THAN WE THINK. One shuffle per layer gave values from
    0.354 to 0.664 -- so a single 0.63 means little. Fix: 300 permutations of
    the source labels, per layer, giving an empirical p-value.

 3. IT IS JUST DURATION (or loudness). Distressed calls are longer and louder
    in most species; if a 4-feature acoustic vector transfers as well as a
    768-d embedding, the "shared structure" is trivial and already known.
    Fix: run the identical pipeline on {log duration, energy, centroid, ZCR}.

Also reported: per-individual transfer accuracy in the target species. If the
effect only exists for two loud dogs, it is not a species-level claim.
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


def fit_predict(Xa, ya, Xb, seed=0):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced",
                              random_state=seed).fit(Xa, ya).predict(Xb)


def zscore_within(X, sp):
    Z = X.copy()
    for s in np.unique(sp):
        m = sp == s
        Z[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-8)
    return Z


def within_cv(X, y, indiv):
    """leave-one-individual-out balanced accuracy, for layer selection"""
    from sklearn.metrics import balanced_accuracy_score
    yp = np.empty_like(y)
    for g in np.unique(indiv):
        te = indiv == g
        if len(np.unique(y[~te])) < 2:
            yp[te] = 0; continue
        yp[te] = fit_predict(X[~te], y[~te], X[te])
    return balanced_accuracy_score(y, yp)


def acoustic_features(rows):
    """4 interpretable features: log duration, RMS energy, spectral centroid, ZCR."""
    from scipy.signal import stft
    F = []
    for r in rows:
        x = r["wav"]
        f, _, Z = stft(x, fs=16000, nperseg=400, noverlap=240, nfft=512)
        P = np.abs(Z) ** 2
        cent = (P * f[:, None]).sum(0) / (P.sum(0) + 1e-12)
        zcr = np.mean(np.abs(np.diff(np.sign(x))) > 0)
        F.append([np.log(len(x) / 16000 + 1e-3), np.log(np.sqrt((x ** 2).mean()) + 1e-8),
                  cent.mean(), zcr])
    return np.array(F, dtype=np.float32)


def main():
    from sklearn.metrics import balanced_accuracy_score as bacc
    import cross_species as cs

    rows = cs.load_all()
    sp = np.array([r["species"] for r in rows])
    y = np.array([r["pol"] for r in rows])
    ind = np.array([r["indiv"] for r in rows])
    E = np.load(os.path.join(OUT, "emb_wavlm.npy")).astype(np.float32)
    c, d = sp == "cat", sp == "dog"
    print(f"{len(rows)} clips · cat {c.sum()} · dog {d.sum()} · layers {E.shape[0]}\n")

    rng = np.random.default_rng(0)
    report = {}

    # ---------------------------------------------------------- 1. per layer, with permutation null
    print("=== per-layer transfer with permutation null (300 perms) ===")
    print(f"{'L':>3} {'c->d':>6} {'p':>6}   {'d->c':>6} {'p':>6}   {'null mean±sd':>16}")
    layers = []
    for li in range(E.shape[0]):
        Z = zscore_within(E[li], sp)
        cd = bacc(y[d], fit_predict(Z[c], y[c], Z[d]))
        dc = bacc(y[c], fit_predict(Z[d], y[d], Z[c]))
        null_cd, null_dc = [], []
        for _ in range(N_PERM):
            null_cd.append(bacc(y[d], fit_predict(Z[c], rng.permutation(y[c]), Z[d])))
            null_dc.append(bacc(y[c], fit_predict(Z[d], rng.permutation(y[d]), Z[c])))
        null_cd, null_dc = np.array(null_cd), np.array(null_dc)
        p_cd = float((null_cd >= cd).mean()); p_dc = float((null_dc >= dc).mean())
        layers.append({"layer": li, "cat_to_dog": cd, "dog_to_cat": dc,
                       "p_cat_to_dog": p_cd, "p_dog_to_cat": p_dc,
                       "null_cd_mean": float(null_cd.mean()), "null_cd_sd": float(null_cd.std()),
                       "null_dc_mean": float(null_dc.mean()), "null_dc_sd": float(null_dc.std()),
                       "cat_within": within_cv(Z[c], y[c], ind[c]),
                       "dog_within": within_cv(Z[d], y[d], ind[d])})
        print(f"{li:>3} {cd:>6.3f} {p_cd:>6.3f}   {dc:>6.3f} {p_dc:>6.3f}   "
              f"{null_cd.mean():.3f}±{null_cd.std():.3f} / {null_dc.mean():.3f}±{null_dc.std():.3f}")
    report["layers"] = layers

    # ---------------------------------------------------------- 2. selection-free summaries
    mean_cd = float(np.mean([l["cat_to_dog"] for l in layers]))
    mean_dc = float(np.mean([l["dog_to_cat"] for l in layers]))
    mean_null_cd = float(np.mean([l["null_cd_mean"] for l in layers]))
    mean_null_dc = float(np.mean([l["null_dc_mean"] for l in layers]))
    print(f"\nmean over ALL layers (no selection): cat->dog {mean_cd:.3f} "
          f"(null {mean_null_cd:.3f})   dog->cat {mean_dc:.3f} (null {mean_null_dc:.3f})")

    # nested: pick layer by source-species within-CV only
    li_c = max(layers, key=lambda l: l["cat_within"])["layer"]
    li_d = max(layers, key=lambda l: l["dog_within"])["layer"]
    nested_cd = layers[li_c]["cat_to_dog"]; nested_dc = layers[li_d]["dog_to_cat"]
    print(f"nested selection (layer chosen on SOURCE within-CV, never on transfer):")
    print(f"  cat->dog  layer {li_c} chosen on cat CV  -> {nested_cd:.3f}  "
          f"(p={layers[li_c]['p_cat_to_dog']:.3f})")
    print(f"  dog->cat  layer {li_d} chosen on dog CV  -> {nested_dc:.3f}  "
          f"(p={layers[li_d]['p_dog_to_cat']:.3f})")
    report["mean_all_layers"] = {"cat_to_dog": mean_cd, "dog_to_cat": mean_dc,
                                 "null_cd": mean_null_cd, "null_dc": mean_null_dc}
    report["nested"] = {"layer_cat": li_c, "layer_dog": li_d,
                        "cat_to_dog": nested_cd, "dog_to_cat": nested_dc,
                        "p_cat_to_dog": layers[li_c]["p_cat_to_dog"],
                        "p_dog_to_cat": layers[li_d]["p_dog_to_cat"]}

    # ---------------------------------------------------------- 3. trivial-feature control
    print("\n=== control: 4 hand-crafted acoustic features (dur, energy, centroid, ZCR) ===")
    A = acoustic_features(rows)
    Za = zscore_within(A, sp)
    a_cd = bacc(y[d], fit_predict(Za[c], y[c], Za[d]))
    a_dc = bacc(y[c], fit_predict(Za[d], y[d], Za[c]))
    n_cd = np.array([bacc(y[d], fit_predict(Za[c], rng.permutation(y[c]), Za[d])) for _ in range(N_PERM)])
    n_dc = np.array([bacc(y[c], fit_predict(Za[d], rng.permutation(y[d]), Za[c])) for _ in range(N_PERM)])
    print(f"  cat->dog {a_cd:.3f} (p={float((n_cd>=a_cd).mean()):.3f})   "
          f"dog->cat {a_dc:.3f} (p={float((n_dc>=a_dc).mean()):.3f})")
    print(f"  within: cat {within_cv(Za[c], y[c], ind[c]):.3f}  dog {within_cv(Za[d], y[d], ind[d]):.3f}")
    report["acoustic_control"] = {"cat_to_dog": a_cd, "dog_to_cat": a_dc,
                                  "p_cd": float((n_cd >= a_cd).mean()),
                                  "p_dc": float((n_dc >= a_dc).mean())}

    # which raw feature actually separates?
    names = ["log_dur", "log_energy", "centroid", "zcr"]
    print("\n  per-feature separation (mean of negative minus mean of positive, in SD):")
    for j, nm in enumerate(names):
        line = f"    {nm:<11s}"
        for s, m in [("cat", c), ("dog", d)]:
            v = A[m, j]; v = (v - v.mean()) / (v.std() + 1e-8)
            line += f"  {s} {v[y[m]==1].mean() - v[y[m]==0].mean():+.2f}"
        print(line)

    # ---------------------------------------------------------- 4. per-individual in the target
    best_li = report["nested"]["layer_cat"]
    Z = zscore_within(E[best_li], sp)
    pred = fit_predict(Z[c], y[c], Z[d])
    print(f"\n=== per-dog accuracy for the cat-trained probe (layer {best_li}) ===")
    yd, indd = y[d], ind[d]
    per = {}
    for g in np.unique(indd):
        m = indd == g
        per[g] = {"n": int(m.sum()), "acc": float((pred[m] == yd[m]).mean()),
                  "n_neg": int((yd[m] == 1).sum())}
        print(f"  {g:<12s} n={m.sum():3d}  acc {per[g]['acc']:.3f}  (neg={per[g]['n_neg']})")
    above = sum(1 for v in per.values() if v["acc"] > 0.5)
    print(f"  -> {above}/{len(per)} dogs above 0.5")
    report["per_dog"] = per

    json.dump(report, open(os.path.join(OUT, "validation.json"), "w"), indent=1)
    print(f"\nwrote {OUT}/validation.json")


if __name__ == "__main__":
    main()
