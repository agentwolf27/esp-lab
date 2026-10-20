"""
Does normalising pitch by body size make animal calls comparable across species?

Pipeline
  1. body-mass assignment (documented assumptions)
  2. fit log10(F0) ~ log10(mass) -- clip level, mass-group level, species level,
     and WITHIN each species
  3. build pitch channels: raw / allometric residual (all-species law) /
     allometric residual (law fitted WITHOUT the test species) /
     within-species z / within-species mean-centred
  4. cross-species affect transfer, 6 directed pairs, balanced accuracy,
     within-group label permutation nulls (200 draws)
"""
from __future__ import annotations
import json, os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bacc

OUT = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(0)
NPERM = 200
SPECIES = ["cat", "dog", "pig"]

# ------------------------------------------------------------------- masses
# Cats: no per-animal mass in CatMeows. Breed x sex point estimates from
# breed-standard adult weights (European/domestic shorthair vs Maine Coon).
CAT_MASS = {("EU", "F"): 4.0, ("EU", "M"): 5.0, ("MC", "F"): 5.5, ("MC", "M"): 8.0}
# Pigs: no mass in the Soundwel key. Midpoints of standard production stages.
PIG_MASS = {"Piglet": 5.0, "Weaner": 20.0, "GrFinishing": 75.0}
# Dogs: per-individual, measured, from annotations.csv.

REF_EXP = {
    "isometric_theory": -1.0 / 3,
    "bowling2017_primates_massconv": -2.456 / 3,
    "bowling2017_carnivores_massconv": -1.004 / 3,
}


def load():
    d = pd.read_csv(os.path.join(OUT, "features.csv"))
    d = d[np.isfinite(d.pol)].copy()
    d["pol"] = d.pol.astype(int)
    m = np.full(len(d), np.nan)
    c = d.species.values == "cat"
    sx = np.array([str(s)[0] if isinstance(s, str) else "F" for s in d.sex.values])
    for i in np.flatnonzero(c):
        m[i] = CAT_MASS[(d.breed.values[i], sx[i])]
    g = d.species.values == "dog"
    m[g] = d.mass_kg.values[g]
    p = d.species.values == "pig"
    m[p] = [PIG_MASS[a] for a in d.age_cat.values[p]]
    d["mass_kg"] = m
    d["mass_group"] = np.where(c, "cat_" + d.breed.astype(str) + "_" + sx,
                      np.where(g, d.group, "pig_" + d.age_cat.astype(str)))
    d["log_mass"] = np.log10(d.mass_kg)
    d["log_f0"] = np.log10(d.f0_med)
    d["log_dur"] = np.log10(d.dur_call)
    d["log_durfile"] = np.log10(d.dur_file)
    return d


def ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    k = np.isfinite(x) & np.isfinite(y)
    x, y = x[k], y[k]
    n = len(x)
    if n < 3 or x.std() < 1e-12:
        return dict(n=int(n), slope=np.nan, intercept=np.nan, r2=np.nan, se=np.nan, p=np.nan)
    A = np.c_[np.ones(n), x]
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    yh = A @ b
    ss = ((y - yh) ** 2).sum()
    r2 = 1 - ss / max(1e-30, ((y - y.mean()) ** 2).sum())
    se = np.sqrt(ss / max(1, n - 2) / max(1e-30, ((x - x.mean()) ** 2).sum()))
    from scipy import stats
    t = b[1] / se if se > 0 else np.nan
    pv = float(2 * stats.t.sf(abs(t), n - 2)) if np.isfinite(t) else np.nan
    return dict(n=int(n), slope=float(b[1]), intercept=float(b[0]), r2=float(r2),
                se=float(se), p=pv)


def group_fit(d, species=None):
    """OLS of median log10 F0 on log10 mass, one point per mass group."""
    q = d if species is None else d[d.species.isin(species)]
    q = q[np.isfinite(q.log_f0)]
    gg = q.groupby("mass_group").agg(log_mass=("log_mass", "first"),
                                     log_f0=("log_f0", "median"),
                                     species=("species", "first"), n=("log_f0", "size"))
    gg = gg[gg.n >= 5]
    r = ols(gg.log_mass, gg.log_f0)
    r["groups"] = gg.reset_index().to_dict("records")
    return r


# ------------------------------------------------------------ pitch channels
def pitch_channels(d, test_species):
    """Return dict name -> pitch vector, for a given held-out test species."""
    lf, lm, sp = d.log_f0.values, d.log_mass.values, d.species.values
    ch = {"raw": lf.copy()}
    law_all = group_fit(d)
    ch["allo_all"] = lf - (law_all["intercept"] + law_all["slope"] * lm)
    others = [s for s in SPECIES if s != test_species]
    law_lo = group_fit(d, species=others)
    ch["allo_loso"] = lf - (law_lo["intercept"] + law_lo["slope"] * lm)
    z = np.empty_like(lf); c = np.empty_like(lf)
    for s in SPECIES:
        k = sp == s
        mu, sd = np.nanmean(lf[k]), np.nanstd(lf[k])
        z[k] = (lf[k] - mu) / (sd + 1e-12)
        c[k] = lf[k] - mu
    ch["zspec"] = z
    ch["center"] = c
    return ch, law_all, law_lo


def zwithin(v, sp):
    o = np.empty_like(v)
    for s in np.unique(sp):
        k = sp == s
        o[k] = (v[k] - np.nanmean(v[k])) / (np.nanstd(v[k]) + 1e-12)
    return o


def fit_predict(Xa, ya, Xb):
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-12          # scaler fitted on TRAIN only
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    clf.fit((Xa - mu) / sd, ya)
    return clf.predict((Xb - mu) / sd)


def perm_within_group(y, groups, rng):
    yp = y.copy()
    for g in np.unique(groups):
        k = np.flatnonzero(groups == g)
        yp[k] = rng.permutation(y[k])
    return yp


def loio(X, y, groups):
    yp = np.empty_like(y)
    for g in np.unique(groups):
        te = groups == g
        if len(np.unique(y[~te])) < 2 or te.sum() == 0:
            yp[te] = y[~te][0] if (~te).sum() else 0
            continue
        yp[te] = fit_predict(X[~te], y[~te], X[te])
    return yp


def main():
    d = load()
    report = {}

    # ---------------------------------------------------------- 1. coverage
    cov = {}
    for s in SPECIES:
        q = d[d.species == s]
        cov[s] = dict(
            n_affect_clips=int(len(q)),
            f0_success=float(np.isfinite(q.f0_med).mean()),
            f0_success_positive=float(np.isfinite(q[q.pol == 0].f0_med).mean()),
            f0_success_negative=float(np.isfinite(q[q.pol == 1].f0_med).mean()),
            n_after_drop=int(np.isfinite(q.f0_med).sum()),
            f0_median_hz=float(np.nanmedian(q.f0_med)),
            f0_iqr_hz=[float(np.nanpercentile(q.f0_med, 25)), float(np.nanpercentile(q.f0_med, 75))],
            dur_call_median_s=float(np.nanmedian(q.dur_call)),
            dur_file_median_s=float(np.nanmedian(q.dur_file)),
            mass_kg=sorted(set(np.round(q.mass_kg.values, 1).tolist())),
            n_groups=int(q.group.nunique()),
            n_mass_groups=int(q.mass_group.nunique()),
        )
    report["coverage"] = cov

    # validation of pig F0 against the corpus's own published F0Mean
    fall = pd.read_csv(os.path.join(OUT, "features.csv"))
    pg = fall[(fall.species == "pig") & np.isfinite(fall.f0_med) & np.isfinite(fall.key_f0mean)]
    report["pig_f0_validation"] = dict(
        n=int(len(pg)),
        log_log_r=float(np.corrcoef(np.log(pg.f0_med), np.log(pg.key_f0mean))[0, 1]),
        median_ratio_mine_over_published=float(np.median(pg.f0_med / pg.key_f0mean)),
        frac_over_1p6x=float(np.mean(pg.f0_med / pg.key_f0mean > 1.6)),
        dur_call_vs_published_dur_r=float(np.corrcoef(
            fall[(fall.species == "pig") & np.isfinite(fall.key_dur)].dur_call,
            fall[(fall.species == "pig") & np.isfinite(fall.key_dur)].key_dur)[0, 1]),
        note="my uniform pyin vs Soundwel key F0Mean (Praat/Avisoft, expert-tuned per corpus)")

    # formant dispersion degeneracy
    fd = {}
    for s in SPECIES:
        q = fall[fall.species == s]
        fd[s] = dict(Df_order12=float(np.nanmedian(q.formant_disp)),
                     nF_order12=float(np.nanmedian(q.n_formants)),
                     Df_order8=float(np.nanmedian(q.formant_disp_o8)),
                     nF_order8=float(np.nanmedian(q.n_formants_o8)))
    report["formant_dispersion_diagnostic"] = fd

    # ------------------------------------------------------- 2. allometry
    d = d[np.isfinite(d.log_f0) & np.isfinite(d.log_dur) & np.isfinite(d.log_rms)].copy()
    print("analysis set:", len(d), {s: int((d.species == s).sum()) for s in SPECIES})

    allo = {}
    allo["group_level_all"] = group_fit(d)
    w = np.zeros(len(d))
    for s in SPECIES:
        k = (d.species == s).values
        w[k] = 1.0 / k.sum()
    # clip level, species-balanced via replication-free weighted OLS
    Xc = np.c_[np.ones(len(d)), d.log_mass.values]
    W = np.diag(w / w.sum())
    b = np.linalg.lstsq(Xc.T @ W @ Xc, Xc.T @ W @ d.log_f0.values, rcond=None)[0]
    yh = Xc @ b
    r2w = 1 - (w * (d.log_f0.values - yh) ** 2).sum() / (w * (d.log_f0.values - np.average(d.log_f0.values, weights=w)) ** 2).sum()
    allo["clip_level_species_balanced"] = dict(n=int(len(d)), slope=float(b[1]),
                                               intercept=float(b[0]), r2=float(r2w))
    allo["clip_level_unweighted"] = ols(d.log_mass, d.log_f0)
    sm = d.groupby("species").agg(lm=("log_mass", "median"), lf=("log_f0", "median"))
    allo["species_level_3pts"] = ols(sm.lm, sm.lf)
    for s in SPECIES:
        q = d[d.species == s]
        allo[f"within_{s}_clip"] = ols(q.log_mass, q.log_f0)
        allo[f"within_{s}_group"] = group_fit(d, species=[s])
    allo["published_reference_exponents_on_log_mass"] = REF_EXP
    allo["note"] = ("Bowling et al. 2017 regressed log F0 on log body LENGTH "
                    "(primates beta=-2.456, R2=0.741; carnivores beta=-1.004, R2=0.443). "
                    "Converted to a mass exponent assuming isometry (M ~ L^3) by dividing by 3.")
    report["allometry"] = allo
    LAW = allo["group_level_all"]
    print(f"group-level law: log10 F0 = {LAW['intercept']:.3f} {LAW['slope']:+.3f}*log10 M  "
          f"R2={LAW['r2']:.3f} n={LAW['n']} p={LAW['p']:.2g}")

    # ------------------------------------------------------- 3. transfer
    sp = d.species.values
    y = d.pol.values
    grp = d.group.values
    ldur_z, lrms_z = zwithin(d.log_dur.values, sp), zwithin(d.log_rms.values, sp)
    ldur_r, lrms_r = d.log_dur.values, d.log_rms.values
    VARIANTS = ["raw", "allo_all", "allo_loso", "zspec", "center"]

    results = {"A_pitch_isolated": {}, "B_whole_vector": {}}
    laws_loso = {}
    for framing in ["A_pitch_isolated", "B_whole_vector"]:
        for v in VARIANTS:
            results[framing][v] = {}

    for te in SPECIES:
        ch, law_all, law_lo = pitch_channels(d, te)
        laws_loso[te] = {k: law_lo[k] for k in ("n", "slope", "intercept", "r2", "p")}
        for tr in SPECIES:
            if tr == te:
                continue
            a, b_ = sp == tr, sp == te
            for v in VARIANTS:
                p = ch[v]
                for framing, (du, rm) in [("A_pitch_isolated", (ldur_z, lrms_z)),
                                          ("B_whole_vector", (ldur_r, lrms_r))]:
                    if framing == "B_whole_vector" and v in ("zspec",):
                        du2, rm2 = ldur_z, lrms_z        # "z-score everything" variant
                    else:
                        du2, rm2 = du, rm
                    X = np.c_[p, du2, rm2]
                    obs = bacc(y[b_], fit_predict(X[a], y[a], X[b_]))
                    rng = np.random.default_rng(abs(hash((tr, te, v, framing))) % (2**32))
                    null = np.array([bacc(y[b_], fit_predict(
                        X[a], perm_within_group(y[a], grp[a], rng), X[b_]))
                        for _ in range(NPERM)])
                    results[framing][v][f"{tr}->{te}"] = dict(
                        bacc=float(obs), null_mean=float(null.mean()),
                        null_p95=float(np.percentile(null, 95)),
                        p=float((1 + (null >= obs).sum()) / (NPERM + 1)),
                        n_train=int(a.sum()), n_test=int(b_.sum()))
            print(f"  {tr}->{te} done", flush=True)
    report["laws_loso"] = laws_loso

    # within-species reference (leave-one-individual/lab-out)
    within = {}
    ch, _, _ = pitch_channels(d, "cat")   # for within-species, all-species law is fine
    for v in VARIANTS:
        within[v] = {}
        for s in SPECIES:
            k = sp == s
            X = np.c_[ch[v][k], ldur_z[k], lrms_z[k]]
            within[v][s] = float(bacc(y[k], loio(X, y[k], grp[k])))
    report["within_species_loio_framingA"] = within

    # per-feature separation, negative minus positive, in SD units (sanity vs prior work)
    sep = {}
    for s in SPECIES:
        k = sp == s
        q = {}
        for nm, vv in [("log_f0", d.log_f0.values), ("log_dur_call", d.log_dur.values),
                       ("log_dur_file", d.log_durfile.values), ("log_rms", d.log_rms.values),
                       ("centroid", d.centroid.values), ("zcr", d.zcr.values)]:
            a1, a0 = vv[k & (y == 1)], vv[k & (y == 0)]
            q[nm] = float((np.nanmean(a1) - np.nanmean(a0)) / (np.nanstd(vv[k]) + 1e-12))
        sep[s] = q
    report["neg_minus_pos_SD"] = sep

    report["config"] = dict(nperm=NPERM, f0_ranges=dict(cat=[180, 1500], dog=[110, 2000], pig=[40, 1500]),
                            cat_mass_kg={f"{k[0]}_{k[1]}": v for k, v in CAT_MASS.items()},
                            pig_mass_kg=PIG_MASS,
                            unit="focal call segment (loudest contiguous energetic run)")
    json.dump(report, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)
    d.to_csv(os.path.join(OUT, "analysis_set.csv"), index=False)

    print("\n=== TRANSFER (framing A: duration & energy within-species z-scored) ===")
    pairs = [f"{a}->{b}" for a in SPECIES for b in SPECIES if a != b]
    print(f"{'pair':<10}" + "".join(f"{v:>12}" for v in VARIANTS))
    for pr in pairs:
        print(f"{pr:<10}" + "".join(
            f"{results['A_pitch_isolated'][v][pr]['bacc']:>9.3f}"
            f"{'*' if results['A_pitch_isolated'][v][pr]['p']<0.05 else ' '}  " for v in VARIANTS))
    print(f"{'MEAN':<10}" + "".join(
        f"{np.mean([results['A_pitch_isolated'][v][p]['bacc'] for p in pairs]):>9.3f}   " for v in VARIANTS))
    print("\n=== TRANSFER (framing B: duration & energy raw) ===")
    for pr in pairs:
        print(f"{pr:<10}" + "".join(
            f"{results['B_whole_vector'][v][pr]['bacc']:>9.3f}"
            f"{'*' if results['B_whole_vector'][v][pr]['p']<0.05 else ' '}  " for v in VARIANTS))
    print(f"{'MEAN':<10}" + "".join(
        f"{np.mean([results['B_whole_vector'][v][p]['bacc'] for p in pairs]):>9.3f}   " for v in VARIANTS))
    print("\nwithin-species LOIO (framing A):")
    for v in VARIANTS:
        print(f"  {v:<10}" + "  ".join(f"{s}={within[v][s]:.3f}" for s in SPECIES))
    report["transfer"] = results
    json.dump(report, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
