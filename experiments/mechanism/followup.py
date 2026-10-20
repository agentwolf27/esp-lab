"""Follow-up on task 3: characterise the one component that passed, check it is
not carried by a handful of animals, cross-check the two decompositions against
each other, and report per-species selectivity counts + the balanced-dictionary
near-misses (a bare "0 passed" hides whether it was close)."""
from __future__ import annotations

import json, os, sys, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import D, OUT, load, zgroup, unit, affect_dir, perm_within
from task3 import ranks, auc_from_ranks, passes, HI, LO, NCOMP, ALPHA, LAYER, SPECIES


def within_group_auc(v, y, g, min_n=6):
    """n-weighted mean of per-group AUCs (groups holding both classes).
    If the effect is real it survives being computed inside each animal."""
    num = den = 0.0
    per = {}
    for s in np.unique(g):
        m = g == s
        if m.sum() < min_n or len(np.unique(y[m])) < 2:
            continue
        a = float(auc_from_ranks(ranks(v[m][:, None]), y[m])[0])
        per[str(s)] = round(a, 3)
        num += a * m.sum(); den += m.sum()
    return (float(num / den) if den else float("nan")), per


def main():
    from sklearn.decomposition import MiniBatchDictionaryLearning, FastICA
    from scipy.stats import spearmanr

    d = load([LAYER])
    Zs = {s: zgroup(d[s]["X"][LAYER], d[s]["zgrp"]) for s in SPECIES}
    Z = np.vstack([Zs[s] for s in SPECIES]).astype(np.float64)
    n = [d[s]["n"] for s in SPECIES]; off = np.cumsum([0] + n)
    idx = {s: np.arange(off[i], off[i + 1]) for i, s in enumerate(SPECIES)}
    ys = {s: d[s]["y"] for s in SPECIES}; grp = {s: d[s]["grp"] for s in SPECIES}

    dl = MiniBatchDictionaryLearning(
        n_components=NCOMP, alpha=ALPHA, batch_size=256, max_iter=300, fit_algorithm="cd",
        transform_algorithm="lasso_lars", transform_alpha=ALPHA, positive_code=True,
        random_state=0, n_jobs=1).fit(Z)
    Cd = dl.transform(Z)
    AUC = np.vstack([auc_from_ranks(ranks(Cd[idx[s]]), ys[s]) for s in SPECIES])
    sel, _, _ = passes(AUC)
    out = {"passing": [int(x) for x in sel]}

    # per-species counts, with the chance expectation from within-group permutation
    rng = np.random.default_rng(21)
    cnt = {s: int(((AUC[i] > HI) | (AUC[i] < LO)).sum()) for i, s in enumerate(SPECIES)}
    nullc = {s: [] for s in SPECIES}
    Rk = {s: ranks(Cd[idx[s]]) for s in SPECIES}
    for _ in range(200):
        for i, s in enumerate(SPECIES):
            a = auc_from_ranks(Rk[s], perm_within(ys[s], grp[s], rng))
            nullc[s].append(int(((a > HI) | (a < LO)).sum()))
    out["per_species_selective_counts"] = {
        s: {"observed": cnt[s], "null_mean": float(np.mean(nullc[s])),
            "null_sd": float(np.std(nullc[s], ddof=1)), "null_max": int(np.max(nullc[s])),
            "p": float((np.sum(np.array(nullc[s]) >= cnt[s]) + 1) / 201)} for s in SPECIES}
    print("selective components per species (AUC>0.60 or <0.40), 192 atoms:")
    for s in SPECIES:
        o = out["per_species_selective_counts"][s]
        print(f"  {s:>4} observed {o['observed']:3d}   null {o['null_mean']:.2f}+-{o['null_sd']:.2f} "
              f"(max {o['null_max']})   p={o['p']:.3f}")

    # how selective does the BEST atom get in each species? (pigs' ceiling matters:
    # 0 passing could mean "no signal" or "signal spread over many atoms")
    out["best_atom_per_species"] = {}
    for i, s in enumerate(SPECIES):
        b = int(np.argmax(np.abs(AUC[i] - 0.5)))
        out["best_atom_per_species"][s] = {"component": b, "auc": float(AUC[i, b]),
                                           "auc_other_species": {t: float(AUC[j, b])
                                                                 for j, t in enumerate(SPECIES)}}
        print(f"  best single atom for {s}: #{b} AUC {AUC[i, b]:.3f} "
              f"(other species: " + ", ".join(f"{t} {AUC[j, b]:.3f}"
                                              for j, t in enumerate(SPECIES) if t != s) + ")")

    # ---- the one passing component, examined per animal
    out["component_detail"] = {}
    for c in sel:
        rec = {"auc": {}, "within_animal_auc": {}, "per_animal": {}, "n_animals_scored": {}}
        for i, s in enumerate(SPECIES):
            v = Cd[idx[s]][:, c]
            rec["auc"][s] = float(AUC[i, c])
            wa, per = within_group_auc(v, ys[s], grp[s])
            rec["within_animal_auc"][s] = wa
            rec["per_animal"][s] = per
            rec["n_animals_scored"][s] = len(per)
        # what the atom itself looks like next to the task-1 affect directions
        atom = dl.components_[c]
        rec["cos_atom_affect_dir"] = {s: float(np.dot(unit(atom), affect_dir(Zs[s], ys[s])))
                                      for s in SPECIES}
        # per-context activation (is it a context detector rather than an affect one?)
        rec["mean_activation_by_context"] = {}
        for s in SPECIES:
            v = Cd[idx[s]][:, c]; ctx = d[s]["ctx"]
            rec["mean_activation_by_context"][s] = {
                str(k): round(float(v[ctx == k].mean()), 3) for k in np.unique(ctx)}
        out["component_detail"][int(c)] = rec
        print(f"\ncomponent {c}: AUC " +
              " ".join(f"{s} {rec['auc'][s]:.3f}" for s in SPECIES))
        print("  AUC computed WITHIN each animal (n-weighted): " +
              " ".join(f"{s} {rec['within_animal_auc'][s]:.3f} (n={rec['n_animals_scored'][s]})"
                       for s in SPECIES))
        print(f"  cos(atom, affect direction): " +
              " ".join(f"{s} {rec['cos_atom_affect_dir'][s]:+.3f}" for s in SPECIES))
        for s in ["cat", "dog"]:
            print(f"  per-{s} AUC: {rec['per_animal'][s]}")
        print(f"  pig mean activation by context: {rec['mean_activation_by_context']['pig']}")

    # ---- cross-check the two decompositions
    ica = FastICA(n_components=64, random_state=0, max_iter=500, whiten="unit-variance")
    S = ica.fit_transform(Z)
    AUCi = np.vstack([auc_from_ranks(ranks(S[idx[s]]), ys[s]) for s in SPECIES])
    seli, _, _ = passes(AUCi)
    out["ica_passing"] = [int(x) for x in seli]
    out["ica_vs_dict"] = {}
    for c in sel:
        for ci in seli:
            r = float(spearmanr(Cd[:, c], S[:, ci]).statistic)
            rp = float(np.corrcoef(Cd[:, c], S[:, ci])[0, 1])
            # are they the same direction in embedding space? the ICA ACTIVATION is
            # read out by the unmixing row components_[ci] (mixing_ is the pattern, not
            # the readout), so that is the vector to compare against.
            wi = unit(ica.components_[ci])
            cosad = float(np.dot(unit(dl.components_[c]), wi))
            cosaf = {s: float(np.dot(wi, affect_dir(Zs[s], ys[s]))) for s in SPECIES}
            out["ica_vs_dict"][f"dict{c}_vs_ica{ci}"] = {
                "spearman": r, "pearson": rp, "cos_dictatom_vs_ica_readout": cosad,
                "ica_auc": {s: float(AUCi[i, ci]) for i, s in enumerate(SPECIES)},
                "cos_ica_readout_affect_dir": cosaf}
            print(f"\ndict comp {c} vs ICA comp {ci}: spearman {r:+.3f}  pearson {rp:+.3f}  "
                  f"cos(atom, ICA readout) {cosad:+.3f}")
            print("  ICA AUC " + " ".join(f"{s} {AUCi[i, ci]:.3f}" for i, s in enumerate(SPECIES)) +
                  "   cos(ICA readout, affect dir) " +
                  " ".join(f"{s} {cosaf[s]:+.3f}" for s in SPECIES))

    # ---- balanced dictionary: how close were the near-misses?
    rng2 = np.random.default_rng(3)
    keep = np.concatenate([idx["cat"], idx["dog"], rng2.permutation(idx["pig"])[:330]])
    dl2 = MiniBatchDictionaryLearning(
        n_components=NCOMP, alpha=ALPHA, batch_size=256, max_iter=300, fit_algorithm="cd",
        transform_algorithm="lasso_lars", transform_alpha=ALPHA, positive_code=True,
        random_state=0, n_jobs=1).fit(Z[keep])
    Cb = dl2.transform(Z)
    AUCb = np.vstack([auc_from_ranks(ranks(Cb[idx[s]]), ys[s]) for s in SPECIES])
    dev = np.abs(AUCb - 0.5)
    second = np.sort(dev, axis=0)[-2]          # 2nd-largest deviation across species
    top = np.argsort(second)[::-1][:5]
    out["balanced_near_misses"] = [{"component": int(c),
                                    "auc": [round(float(AUCb[i, c]), 3) for i in range(3)]}
                                   for c in top]
    print("\nspecies-balanced dictionary, 5 best 'shared' candidates (cat/dog/pig AUC):")
    for r in out["balanced_near_misses"]:
        print(f"  comp {r['component']:>3}  {r['auc']}")

    json.dump(out, open(os.path.join(OUT, "followup.json"), "w"), indent=1)
    print(f"\nwrote {OUT}/followup.json")


if __name__ == "__main__":
    main()
