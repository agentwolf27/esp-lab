"""Tasks 1 & 2.

1. Per-species affect direction w_s = normalize(mean_neg - mean_pos) on
   within-species z-scored embeddings (pigs z-scored within recording team).
   Pairwise cosines + a label-permutation null (200 draws, labels shuffled
   WITHIN animal for cat/dog and WITHIN team for pigs). Repeated after
   regressing log-duration out of every embedding dimension, per species.
   Also cos(w_s, d_s) with d_s the OLS coefficient vector on log-duration.

2. Transfer as a 1-D projection: probe weights w_A from species A, project
   species B, AUC for B's affect labels, and Spearman of that projection with
   B's duration / energy / centroid.
"""
from __future__ import annotations

import json, os, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (D, OUT, load, zgroup, unit, affect_dir, duration_dir,
                    residualise, perm_within)

LAYERS = [9, 12, 3]
SPECIES = ["cat", "dog", "pig"]
PAIRS = [("cat", "dog"), ("cat", "pig"), ("dog", "pig")]
NPERM = 200
SEED = 11


def cos(a, b):
    return float(np.dot(unit(a), unit(b)))


def group_spearman(x, v, g):
    """Spearman of x vs v computed within each group, n-weighted mean.
    (For pigs the embeddings are team-z-scored, so the global correlation
    mixes in between-team structure; this is the team-controlled version.)"""
    from scipy.stats import spearmanr
    num, den = 0.0, 0
    for s in np.unique(g):
        m = g == s
        if m.sum() < 8:
            continue
        r = spearmanr(x[m], v[m]).statistic
        if np.isfinite(r):
            num += r * m.sum(); den += m.sum()
    return float(num / den) if den else float("nan")


def resid_on(x, C):
    """residualise vector x on columns of C (+intercept)"""
    C = np.column_stack([np.ones(len(x)), C])
    beta, *_ = np.linalg.lstsq(C, x, rcond=None)
    return x - C @ beta


def split_half_reliability(Z, y, grp, rng, n_split=50, by_group=True):
    """How well is w_s estimated at all? Split the data in two, recompute
    w on each half, take the cosine. Averaged over n_split random splits.
    by_group=True splits whole ANIMALS (the honest unit for cat/dog);
    for pigs there is no individual id, so we split calls within team."""
    cs = []
    units = np.unique(grp)
    for _ in range(n_split):
        if by_group and len(units) >= 4:
            perm = rng.permutation(units)
            h1 = np.isin(grp, perm[: len(units) // 2])
        else:                                   # split rows within group
            h1 = np.zeros(len(y), dtype=bool)
            for u in units:
                m = np.flatnonzero(grp == u)
                h1[rng.permutation(m)[: len(m) // 2]] = True
        h2 = ~h1
        if min(len(np.unique(y[h1])), len(np.unique(y[h2]))) < 2:
            continue
        cs.append(cos(affect_dir(Z[h1], y[h1]), affect_dir(Z[h2], y[h2])))
    return float(np.mean(cs)) if cs else float("nan"), len(cs)


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr

    t0 = time.time()
    d = load(LAYERS)
    for s in SPECIES:
        print(f"{s:>4} n={d[s]['n']:5d} neg={int((d[s]['y']==1).sum()):5d} "
              f"pos={int((d[s]['y']==0).sum()):5d} perm-units={len(np.unique(d[s]['grp']))}")
    res = {"n": {s: d[s]["n"] for s in SPECIES},
           "n_perm": NPERM, "layers": LAYERS,
           "perm_unit": {"cat": "individual", "dog": "individual", "pig": "recording team"},
           "zscore_unit": {"cat": "species", "dog": "species", "pig": "recording team"},
           "task1": {}, "task2": {}}

    for li in LAYERS:
        print(f"\n================ LAYER {li} ================")
        Z, R, w, wr, dd, ld = {}, {}, {}, {}, {}, {}
        for s in SPECIES:
            Z[s] = zgroup(d[s]["X"][li], d[s]["zgrp"])
            ld[s] = d[s]["A"][:, 0].astype(np.float64)          # log duration
            R[s] = residualise(Z[s], ld[s])
            w[s] = affect_dir(Z[s], d[s]["y"])
            wr[s] = affect_dir(R[s], d[s]["y"])
            dd[s] = unit(duration_dir(Z[s], ld[s]))

        # ---- observed cosines
        obs = {f"{a}-{b}": cos(w[a], w[b]) for a, b in PAIRS}
        obs_r = {f"{a}-{b}": cos(wr[a], wr[b]) for a, b in PAIRS}
        cwd = {s: cos(w[s], dd[s]) for s in SPECIES}
        cwd_r = {s: cos(wr[s], dd[s]) for s in SPECIES}
        # cosine between the three species' DURATION directions (is duration itself shared?)
        cdd = {f"{a}-{b}": cos(dd[a], dd[b]) for a, b in PAIRS}
        # effect size of the raw (unnormalised) affect contrast, in z units
        eff = {s: float(np.linalg.norm(Z[s][d[s]["y"] == 1].mean(0) - Z[s][d[s]["y"] == 0].mean(0)))
               for s in SPECIES}
        eff_r = {s: float(np.linalg.norm(R[s][d[s]["y"] == 1].mean(0) - R[s][d[s]["y"] == 0].mean(0)))
                 for s in SPECIES}

        # ---- permutation null
        rng = np.random.default_rng(SEED + li)
        null = {k: [] for k in obs}
        null_r = {k: [] for k in obs}
        for _ in range(NPERM):
            wp, wpr = {}, {}
            for s in SPECIES:
                yp = perm_within(d[s]["y"], d[s]["grp"], rng)
                wp[s] = affect_dir(Z[s], yp)
                wpr[s] = affect_dir(R[s], yp)
            for a, b in PAIRS:
                null[f"{a}-{b}"].append(cos(wp[a], wp[b]))
                null_r[f"{a}-{b}"].append(cos(wpr[a], wpr[b]))

        def summarise(o, nl):
            n = np.array(nl)
            return {"cos": o, "null_mean": float(n.mean()), "null_sd": float(n.std(ddof=1)),
                    "p_greater": float((np.sum(n >= o) + 1) / (len(n) + 1)),
                    "p_two_sided": float((np.sum(np.abs(n) >= abs(o)) + 1) / (len(n) + 1)),
                    "z": float((o - n.mean()) / (n.std(ddof=1) + 1e-12))}

        # ---- how well is w_s estimated at all? (split-half, animals held out)
        rng2 = np.random.default_rng(101 + li)
        rel, rel_r, rdur = {}, {}, {}
        for s in SPECIES:
            by = s != "pig"                       # pigs have no individual id
            rel[s] = split_half_reliability(Z[s], d[s]["y"], d[s]["grp"], rng2, by_group=by)[0]
            rel_r[s] = split_half_reliability(R[s], d[s]["y"], d[s]["grp"], rng2, by_group=by)[0]
            # what the affect score itself tracks physically, inside the species
            rdur[s] = {"spearman_score_logdur": float(spearmanr(Z[s] @ w[s], ld[s]).statistic),
                       "spearman_score_logdur_resid": float(spearmanr(R[s] @ wr[s], ld[s]).statistic),
                       "auc_own_axis": float(roc_auc_score(d[s]["y"], Z[s] @ w[s])),
                       "auc_own_axis_resid": float(roc_auc_score(d[s]["y"], R[s] @ wr[s]))}
        # attenuation-corrected cosine: cos / sqrt(rel_a * rel_b)
        disatt = {f"{a}-{b}": (float(obs[f"{a}-{b}"] / np.sqrt(rel[a] * rel[b]))
                               if rel[a] > 0 and rel[b] > 0 else float("nan")) for a, b in PAIRS}
        disatt_r = {f"{a}-{b}": (float(obs_r[f"{a}-{b}"] / np.sqrt(rel_r[a] * rel_r[b]))
                                 if rel_r[a] > 0 and rel_r[b] > 0 else float("nan")) for a, b in PAIRS}

        lay = {"raw": {k: summarise(obs[k], null[k]) for k in obs},
               "dur_residualised": {k: summarise(obs_r[k], null_r[k]) for k in obs_r},
               "cos_w_dur": cwd, "cos_w_dur_after_resid": cwd_r,
               "cos_dur_dur": cdd, "affect_norm_z": eff, "affect_norm_z_resid": eff_r,
               "split_half_reliability": rel, "split_half_reliability_resid": rel_r,
               "cos_disattenuated": disatt, "cos_disattenuated_resid": disatt_r,
               "own_axis": rdur}

        print("  pair      cos      null(mean+-sd)      z      p(>)   | after duration-residualisation")
        for k in obs:
            a, b = lay["raw"][k], lay["dur_residualised"][k]
            print(f"  {k:<8} {a['cos']:+.3f}   {a['null_mean']:+.3f}+-{a['null_sd']:.3f}  "
                  f"{a['z']:+6.2f}  {a['p_greater']:.3f}  |  {b['cos']:+.3f}  "
                  f"{b['null_mean']:+.3f}+-{b['null_sd']:.3f}  {b['z']:+6.2f}  {b['p_greater']:.3f}")
        print("  cos(w_s, duration_dir):  " +
              "  ".join(f"{s} {cwd[s]:+.3f}" for s in SPECIES) +
              "   | after resid: " + "  ".join(f"{s} {cwd_r[s]:+.3f}" for s in SPECIES))
        print("  cos(dur_a, dur_b):       " + "  ".join(f"{k} {v:+.3f}" for k, v in cdd.items()))
        print("  ||mean_neg-mean_pos|| :  " + "  ".join(f"{s} {eff[s]:.2f}" for s in SPECIES) +
              "   | resid: " + "  ".join(f"{s} {eff_r[s]:.2f}" for s in SPECIES))
        print("  split-half reliability of w_s (ceiling for any cosine): " +
              "  ".join(f"{s} {rel[s]:+.3f}" for s in SPECIES))
        print("  attenuation-corrected cos: " + "  ".join(f"{k} {v:+.3f}" for k, v in disatt.items()))
        print("  spearman(own affect score, log-dur): " +
              "  ".join(f"{s} {rdur[s]['spearman_score_logdur']:+.3f}" for s in SPECIES))
        res["task1"][str(li)] = lay

        # ================= task 2: transfer as a 1-D projection =================
        print("\n  --- transfer projection (train A, project B) ---")
        print("  A->B      AUC    AUC|dur   rho_dur  rho_energy  rho_centroid   (rho within-team for pig)")
        t2 = {}
        for a in SPECIES:
            clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced",
                                     random_state=0).fit(Z[a], d[a]["y"])
            wA = clf.coef_[0]
            for b in SPECIES:
                if a == b:
                    continue
                proj = Z[b] @ wA
                yb = d[b]["y"]; Ab = d[b]["A"]; gb = d[b]["zgrp"]
                auc = float(roc_auc_score(yb, proj))
                # projection with log-duration partialled out (within z-scoring unit)
                pr = proj.astype(np.float64).copy()
                for s in np.unique(gb):
                    m = gb == s
                    pr[m] = resid_on(pr[m], Ab[m, 0:1].astype(np.float64))
                auc_pd = float(roc_auc_score(yb, pr))
                rho = {n: float(spearmanr(proj, Ab[:, i]).statistic)
                       for i, n in [(0, "log_dur"), (1, "log_energy"), (2, "centroid"), (3, "zcr")]}
                rho_w = {n: group_spearman(proj, Ab[:, i], gb)
                         for i, n in [(0, "log_dur"), (1, "log_energy"), (2, "centroid"), (3, "zcr")]}
                # same thing but with the MEAN-DIFFERENCE axis of task 1 (not the LR probe),
                # so tasks 1 and 2 are talking about the same vector
                auc_md = float(roc_auc_score(yb, Z[b] @ w[a]))
                t2[f"{a}->{b}"] = {"auc": auc, "auc_dur_partialled": auc_pd,
                                   "auc_meandiff_axis": auc_md,
                                   "spearman": rho, "spearman_within_zunit": rho_w}
                print(f"  {a}->{b:<4} {auc:.3f}   {auc_pd:.3f}  [md {auc_md:.3f}]  "
                      f"{rho['log_dur']:+.3f}    {rho['log_energy']:+.3f}      {rho['centroid']:+.3f}"
                      + (f"      [{rho_w['log_dur']:+.3f} {rho_w['log_energy']:+.3f} {rho_w['centroid']:+.3f}]"
                         if b == "pig" else ""))
        # reference: what do the acoustics alone give in each target
        for b in SPECIES:
            yb = d[b]["y"]; Ab = d[b]["A"]
            t2[f"acoustic_alone_{b}"] = {n: float(roc_auc_score(yb, Ab[:, i]))
                                         for i, n in [(0, "log_dur"), (1, "log_energy"),
                                                      (2, "centroid"), (3, "zcr")]}
        print("  reference AUC of raw acoustics for affect: " +
              "; ".join(f"{b}: dur {t2[f'acoustic_alone_{b}']['log_dur']:.3f} "
                        f"eng {t2[f'acoustic_alone_{b}']['log_energy']:.3f} "
                        f"cen {t2[f'acoustic_alone_{b}']['centroid']:.3f}" for b in SPECIES))
        res["task2"][str(li)] = t2

    json.dump(res, open(os.path.join(OUT, "task12.json"), "w"), indent=1)
    print(f"\nwrote {OUT}/task12.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
