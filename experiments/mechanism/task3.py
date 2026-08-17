"""Task 3: sparse features shared across species.

Fit a sparse dictionary on the POOLED, within-species z-scored layer-9
embeddings (cat 348 + dog 308 + pig 5031 = 5687 vectors, 768-d).
MiniBatchDictionaryLearning, fit_algorithm='cd' (required for positive codes),
transform_algorithm='lasso_lars', positive_code=True.

For every component: per-species affect selectivity = AUC of its activation for
neg vs pos, within species. A component "counts" if selectivity >0.60 or <0.40
in >=2 species WITH THE SAME SIGN. Because 192 components x 3 species is a lot
of tests, the same criterion is run on 200 label permutations (within animal /
within team) to get the expected number of chance passes.

For the passing components: Spearman with duration / energy / centroid, and the
selectivity recomputed after regressing the acoustics out of the activation --
the test of whether the component is duration in disguise.
"""
from __future__ import annotations

import json, os, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import D, OUT, load, zgroup, perm_within

LAYER = 9
SPECIES = ["cat", "dog", "pig"]
NCOMP = 192
ALPHA = 1.0
NPERM = 200
HI, LO = 0.60, 0.40


def resid_cols(x, C, g):
    """residualise x on columns C (+intercept), separately within each group g"""
    out = x.astype(np.float64).copy()
    for s in np.unique(g):
        m = g == s
        M = np.column_stack([np.ones(int(m.sum())), C[m]])
        beta, *_ = np.linalg.lstsq(M, out[m], rcond=None)
        out[m] = out[m] - M @ beta
    return out


def ranks(M):
    """average ranks down each column (ties -> mean rank, matching roc_auc_score)"""
    from scipy.stats import rankdata
    return rankdata(M, axis=0).astype(np.float64)


def auc_from_ranks(Rk, y):
    """Mann-Whitney AUC for every column at once; y==1 is the positive class.
    Ranks do not depend on the labels, so the permutation null is just a matmul."""
    n1 = float((y == 1).sum()); n0 = float((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.full(Rk.shape[1], 0.5)
    s = Rk[y == 1].sum(0)
    return (s - n1 * (n1 + 1) / 2) / (n1 * n0)


def aucs_per_species(Cd, idx, ys):
    """(n_species, n_comp) AUC matrix"""
    out = np.full((len(SPECIES), Cd.shape[1]), 0.5)
    for si, s in enumerate(SPECIES):
        Rk = ranks(Cd[idx[s]])
        a = auc_from_ranks(Rk, ys[s])
        flat = Cd[idx[s]].max(0) - Cd[idx[s]].min(0) < 1e-12
        a[flat] = 0.5
        out[si] = a
    return out


def passes(AUC):
    """components selective (>HI or <LO) in >=2 species with the SAME sign"""
    hi = (AUC > HI).sum(0)
    lo = (AUC < LO).sum(0)
    return np.flatnonzero((hi >= 2) | (lo >= 2)), hi, lo


def main():
    from sklearn.decomposition import MiniBatchDictionaryLearning, FastICA
    from scipy.stats import spearmanr

    t0 = time.time()
    d = load([LAYER])
    Zs = {s: zgroup(d[s]["X"][LAYER], d[s]["zgrp"]) for s in SPECIES}
    Z = np.vstack([Zs[s] for s in SPECIES]).astype(np.float64)
    n = [d[s]["n"] for s in SPECIES]
    off = np.cumsum([0] + n)
    idx = {s: np.arange(off[i], off[i + 1]) for i, s in enumerate(SPECIES)}
    ys = {s: d[s]["y"] for s in SPECIES}
    grp = {s: d[s]["grp"] for s in SPECIES}
    zg = {s: d[s]["zgrp"] for s in SPECIES}
    A = {s: d[s]["A"] for s in SPECIES}
    print(f"pooled {Z.shape}  (cat {n[0]}, dog {n[1]}, pig {n[2]})", flush=True)

    res = {"layer": LAYER, "n_pooled": int(Z.shape[0]), "n_per_species": dict(zip(SPECIES, n)),
           "method": ("sklearn MiniBatchDictionaryLearning, fit_algorithm='cd' "
                      "(positive codes are not supported by the default 'lars' fit), "
                      "transform_algorithm='lasso_lars', positive_code=True"),
           "n_components": NCOMP, "alpha": ALPHA, "n_perm": NPERM,
           "criterion": f"AUC > {HI} or < {LO} in >=2 species, same sign"}

    # ---------------- fit
    t = time.time()
    dl = MiniBatchDictionaryLearning(
        n_components=NCOMP, alpha=ALPHA, batch_size=256, max_iter=300,
        fit_algorithm="cd", transform_algorithm="lasso_lars", transform_alpha=ALPHA,
        positive_code=True, random_state=0, n_jobs=1)
    dl.fit(Z)
    Cd = dl.transform(Z)
    fit_s = time.time() - t
    nnz = (Cd > 1e-8).sum(1)
    recon = 1 - ((Z - Cd @ dl.components_) ** 2).sum() / (Z ** 2).sum()
    print(f"dict learning: {fit_s:.0f}s   active atoms/sample {nnz.mean():.1f}/{NCOMP}   "
          f"variance explained {recon:.3f}", flush=True)
    res["runtime_s"] = round(fit_s, 1)
    res["active_atoms_per_sample"] = float(nnz.mean())
    res["variance_explained"] = float(recon)
    res["frac_nonzero_per_species"] = {s: float((Cd[idx[s]] > 1e-8).mean()) for s in SPECIES}

    # ---------------- observed selectivity
    AUC = aucs_per_species(Cd, idx, ys)
    sel, hi, lo = passes(AUC)
    print(f"\ncomponents passing in >=2 species (same sign): {len(sel)} of {NCOMP}")
    all3 = np.flatnonzero(((AUC > HI).sum(0) == 3) | ((AUC < LO).sum(0) == 3))
    print(f"components passing in ALL THREE species (same sign): {len(all3)}  {list(map(int, all3))}")

    # ---------------- how many would pass by chance?
    rng = np.random.default_rng(7)
    null_counts, null_counts3 = [], []
    Cs = {s: Cd[idx[s]] for s in SPECIES}
    Rk = {s: ranks(Cs[s]) for s in SPECIES}
    flat = {s: (Cs[s].max(0) - Cs[s].min(0) < 1e-12) for s in SPECIES}
    null_auc = {s: np.zeros((NPERM, NCOMP)) for s in SPECIES}
    from sklearn.metrics import roc_auc_score
    for p in range(NPERM):
        Ap = np.full((3, NCOMP), 0.5)
        for si, s in enumerate(SPECIES):
            yp = perm_within(ys[s], grp[s], rng)
            a = auc_from_ranks(Rk[s], yp)
            a[flat[s]] = 0.5
            Ap[si] = a
            null_auc[s][p] = a
        s2, _, _ = passes(Ap)
        null_counts.append(len(s2))
        null_counts3.append(int((((Ap > HI).sum(0) == 3) | ((Ap < LO).sum(0) == 3)).sum()))
        if (p + 1) % 100 == 0:
            print(f"  perm {p+1}/{NPERM}  ({time.time()-t0:.0f}s)", flush=True)
    nc = np.array(null_counts); nc3 = np.array(null_counts3)
    print(f"null passes (>=2 species): mean {nc.mean():.2f} +- {nc.std(ddof=1):.2f}  "
          f"max {nc.max()}   p(count>=obs) = {(np.sum(nc >= len(sel))+1)/(NPERM+1):.3f}")
    print(f"null passes (all 3):       mean {nc3.mean():.2f} +- {nc3.std(ddof=1):.2f}  "
          f"max {nc3.max()}   p(count>=obs) = {(np.sum(nc3 >= len(all3))+1)/(NPERM+1):.3f}")
    res["n_pass_2species"] = int(len(sel))
    res["n_pass_3species"] = int(len(all3))
    res["null_pass_2species"] = {"mean": float(nc.mean()), "sd": float(nc.std(ddof=1)),
                                 "max": int(nc.max()),
                                 "p": float((np.sum(nc >= len(sel)) + 1) / (NPERM + 1))}
    res["null_pass_3species"] = {"mean": float(nc3.mean()), "sd": float(nc3.std(ddof=1)),
                                 "max": int(nc3.max()),
                                 "p": float((np.sum(nc3 >= len(all3)) + 1) / (NPERM + 1))}

    # per-component empirical p (two-sided on |AUC-0.5|), from the same permutations,
    # pooled across components to form the null of max deviation -> conservative
    # (recomputed cheaply: store per-species null AUC spread)
    # ---------------- detail on the passing components
    comps = []
    for c in sorted(set(sel.tolist()) | set(all3.tolist())):
        rec = {"component": int(c), "auc": {s: float(AUC[si, c]) for si, s in enumerate(SPECIES)},
               "frac_active": {s: float((Cs[s][:, c] > 1e-8).mean()) for s in SPECIES},
               "n_species_pass": int(max((AUC[:, c] > HI).sum(), (AUC[:, c] < LO).sum())),
               "sign": "neg-high" if (AUC[:, c] > HI).sum() >= 2 else "pos-high",
               "spearman": {}, "auc_dur_controlled": {}, "auc_acoustics_controlled": {},
               "partial_spearman_affect_given_acoustics": {},
               "p_perm_two_sided": {s: float((np.sum(np.abs(null_auc[s][:, c] - 0.5)
                                                     >= abs(AUC[si, c] - 0.5)) + 1) / (NPERM + 1))
                                    for si, s in enumerate(SPECIES)},
               "null_auc_sd": {s: float(null_auc[s][:, c].std(ddof=1)) for s in SPECIES}}
        for si, s in enumerate(SPECIES):
            v = Cs[s][:, c].astype(np.float64)
            Aa = A[s].astype(np.float64)
            rec["spearman"][s] = {nm: float(spearmanr(v, Aa[:, i]).statistic)
                                  for i, nm in [(0, "log_dur"), (1, "log_energy"),
                                                (2, "centroid"), (3, "zcr")]}
            if v.max() - v.min() < 1e-12:
                rec["auc_dur_controlled"][s] = 0.5
                rec["auc_acoustics_controlled"][s] = 0.5
                rec["partial_spearman_affect_given_acoustics"][s] = 0.0
                continue
            vd = resid_cols(v, Aa[:, 0:1], zg[s])
            va = resid_cols(v, Aa[:, 0:3], zg[s])
            rec["auc_dur_controlled"][s] = float(roc_auc_score(ys[s], vd))
            rec["auc_acoustics_controlled"][s] = float(roc_auc_score(ys[s], va))
            # rank-based partial correlation of activation with affect | acoustics
            from scipy.stats import rankdata
            rv = rankdata(v); ry = rankdata(ys[s])
            RA = np.column_stack([rankdata(Aa[:, i]) for i in range(3)])
            rv_r = resid_cols(rv, RA, zg[s]); ry_r = resid_cols(ry, RA, zg[s])
            den = np.std(rv_r) * np.std(ry_r)
            rec["partial_spearman_affect_given_acoustics"][s] = (
                float(np.mean((rv_r - rv_r.mean()) * (ry_r - ry_r.mean())) / den) if den > 0 else 0.0)
        comps.append(rec)
    res["components"] = comps

    print(f"\n{'comp':>4} {'AUC cat':>8} {'AUC dog':>8} {'AUC pig':>8} | "
          f"{'dur-ctrl (cat/dog/pig)':>26} | {'rho_dur (cat/dog/pig)':>24} | active%")
    for r in comps:
        a = r["auc"]; dc = r["auc_dur_controlled"]; sp = r["spearman"]
        print(f"{r['component']:>4} {a['cat']:>8.3f} {a['dog']:>8.3f} {a['pig']:>8.3f} | "
              f"{dc['cat']:>8.3f} {dc['dog']:>8.3f} {dc['pig']:>7.3f} | "
              f"{sp['cat']['log_dur']:>+7.3f} {sp['dog']['log_dur']:>+7.3f} {sp['pig']['log_dur']:>+7.3f} | "
              f"{100*r['frac_active']['cat']:.0f}/{100*r['frac_active']['dog']:.0f}/"
              f"{100*r['frac_active']['pig']:.0f}")

    # ---------------- robustness: FastICA, and a species-balanced dictionary
    ica = FastICA(n_components=64, random_state=0, max_iter=500, whiten="unit-variance")
    S = ica.fit_transform(Z)
    AUCi = aucs_per_species(S, idx, ys)
    seli, _, _ = passes(AUCi)
    all3i = np.flatnonzero(((AUCi > HI).sum(0) == 3) | ((AUCi < LO).sum(0) == 3))
    print(f"\nFastICA(64) cross-check: {len(seli)} pass in >=2 species, {len(all3i)} in all three")
    res["fastica"] = {"n_components": 64, "n_pass_2species": int(len(seli)),
                      "n_pass_3species": int(len(all3i)),
                      "pass_ids": [int(x) for x in seli],
                      "auc_of_passing": {int(c): [float(AUCi[k, c]) for k in range(3)] for c in seli}}

    rng2 = np.random.default_rng(3)
    keep = np.concatenate([idx["cat"], idx["dog"],
                           rng2.permutation(idx["pig"])[:330]])
    dl2 = MiniBatchDictionaryLearning(
        n_components=NCOMP, alpha=ALPHA, batch_size=256, max_iter=300, fit_algorithm="cd",
        transform_algorithm="lasso_lars", transform_alpha=ALPHA, positive_code=True,
        random_state=0, n_jobs=1).fit(Z[keep])
    Cb = dl2.transform(Z)
    AUCb = aucs_per_species(Cb, idx, ys)
    selb, _, _ = passes(AUCb)
    all3b = np.flatnonzero(((AUCb > HI).sum(0) == 3) | ((AUCb < LO).sum(0) == 3))
    print(f"species-balanced dictionary (cat+dog+330 pigs): {len(selb)} pass in >=2, {len(all3b)} in all three")
    res["balanced_dict"] = {"n_pass_2species": int(len(selb)), "n_pass_3species": int(len(all3b)),
                            "pass_ids": [int(x) for x in selb]}

    np.savez(os.path.join(OUT, "codes.npz"), auc=AUC, codes_summary=Cd[:, sorted(set(sel.tolist()))],
             sel=np.array(sorted(set(sel.tolist())), dtype=int),
             dictionary=dl.components_.astype(np.float32))
    json.dump(res, open(os.path.join(OUT, "task3.json"), "w"), indent=1)
    print(f"\nwrote {OUT}/task3.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
