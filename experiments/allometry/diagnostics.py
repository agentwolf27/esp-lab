"""Decisive controls for the allometric-normalisation claim."""
from __future__ import annotations
import json, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bacc
import analyze as A

OUT = os.path.dirname(os.path.abspath(__file__))
SPECIES = A.SPECIES
NBOOT = 1000


def build(d):
    sp = d.species.values
    return dict(sp=sp, y=d.pol.values, grp=d.group.values,
                dz=A.zwithin(d.log_dur.values, sp), rz=A.zwithin(d.log_rms.values, sp))


def main():
    rep = json.load(open(os.path.join(OUT, "results.json")))
    d = A.load()
    d = d[np.isfinite(d.log_f0) & np.isfinite(d.log_dur) & np.isfinite(d.log_rms)].copy()
    B = build(d)
    sp, y, grp = B["sp"], B["y"], B["grp"]

    # ---- 1. is the allometric residual anything other than species-mean centring?
    eq = {}
    for te in SPECIES:
        ch, law_all, _ = A.pitch_channels(d, te)
        break
    for s in SPECIES:
        k = sp == s
        r_allo, r_cent, lf = ch["allo_all"][k], ch["center"][k], d.log_f0.values[k]
        eq[s] = dict(
            mean_allometric_residual_log10=float(r_allo.mean()),
            observed_over_predicted_F0=float(10 ** r_allo.mean()),
            corr_allo_vs_speciescentred=float(np.corrcoef(r_allo, r_cent)[0, 1]),
            sd_allo_minus_speciescentred_log10=float(np.std(r_allo - r_cent)),
            sd_of_logF0_within_species=float(np.std(lf)))
    rep["allometry_vs_species_centering"] = eq

    # ---- 2. no-pitch baseline + paired cluster bootstrap on Dbacc
    VAR = ["nopitch", "raw", "allo_all", "allo_loso", "zspec", "center"]
    tab, boot = {}, {}
    for te in SPECIES:
        ch, _, _ = A.pitch_channels(d, te)
        ch["nopitch"] = None
        for tr in SPECIES:
            if tr == te:
                continue
            a, b_ = sp == tr, sp == te
            preds = {}
            for v in VAR:
                X = (np.c_[B["dz"], B["rz"]] if v == "nopitch"
                     else np.c_[ch[v], B["dz"], B["rz"]])
                preds[v] = A.fit_predict(X[a], y[a], X[b_])
            key = f"{tr}->{te}"
            tab[key] = {v: float(bacc(y[b_], preds[v])) for v in VAR}
            # cluster bootstrap over TEST groups
            gte = grp[b_]; ug = np.unique(gte); yte = y[b_]
            rng = np.random.default_rng(7)
            acc = {v: [] for v in VAR}
            for _ in range(NBOOT):
                pick = rng.choice(ug, len(ug), replace=True)
                idx = np.concatenate([np.flatnonzero(gte == g) for g in pick])
                if len(np.unique(yte[idx])) < 2:
                    continue
                for v in VAR:
                    acc[v].append(bacc(yte[idx], preds[v][idx]))
            boot[key] = {}
            for v in VAR:
                if v == "raw":
                    continue
                dlt = np.array(acc[v]) - np.array(acc["raw"])
                boot[key][f"{v}_minus_raw"] = dict(
                    delta=float(tab[key][v] - tab[key]["raw"]),
                    ci95=[float(np.percentile(dlt, 2.5)), float(np.percentile(dlt, 97.5))])
            for v in ["allo_all", "allo_loso"]:
                dlt = np.array(acc[v]) - np.array(acc["zspec"])
                boot[key][f"{v}_minus_zspec"] = dict(
                    delta=float(tab[key][v] - tab[key]["zspec"]),
                    ci95=[float(np.percentile(dlt, 2.5)), float(np.percentile(dlt, 97.5))])
    rep["transfer_with_nopitch_baseline"] = tab
    rep["paired_bootstrap_deltas"] = boot

    pairs = list(tab.keys())
    means = {v: float(np.mean([tab[p][v] for p in pairs])) for v in VAR}
    rep["transfer_means_over_6_pairs"] = means

    # ---- 3. sensitivity: pigs measured with the corpus's own published F0Mean
    f = pd.read_csv(os.path.join(OUT, "features.csv"))
    f = f[np.isfinite(f.pol)].copy()
    d2 = A.load()
    d2.loc[d2.species == "pig", "log_f0"] = np.log10(
        f.loc[f.species == "pig", "key_f0mean"].values)
    d2 = d2[np.isfinite(d2.log_f0) & np.isfinite(d2.log_dur) & np.isfinite(d2.log_rms)].copy()
    B2 = build(d2); sp2, y2 = B2["sp"], B2["y"]
    law2 = A.group_fit(d2)
    sens = {"n": {s: int((sp2 == s).sum()) for s in SPECIES},
            "law": {k: law2[k] for k in ("n", "slope", "intercept", "r2", "p")},
            "pig_median_f0_hz": float(10 ** np.median(d2.log_f0.values[sp2 == "pig"])),
            "transfer": {}}
    for te in SPECIES:
        ch2, _, _ = A.pitch_channels(d2, te)
        ch2["nopitch"] = None
        for tr in SPECIES:
            if tr == te:
                continue
            a, b_ = sp2 == tr, sp2 == te
            row = {}
            for v in VAR:
                X = (np.c_[B2["dz"], B2["rz"]] if v == "nopitch"
                     else np.c_[ch2[v], B2["dz"], B2["rz"]])
                row[v] = float(bacc(y2[b_], A.fit_predict(X[a], y2[a], X[b_])))
            sens["transfer"][f"{tr}->{te}"] = row
    sens["means"] = {v: float(np.mean([r[v] for r in sens["transfer"].values()])) for v in VAR}
    rep["sensitivity_published_pig_F0"] = sens

    # ---- 4. sensitivity: file duration instead of focal-call duration
    d3 = d.copy()
    B3 = build(d3); B3["dz"] = A.zwithin(d3.log_durfile.values, sp)
    sens2 = {}
    for te in SPECIES:
        ch3, _, _ = A.pitch_channels(d3, te)
        ch3["nopitch"] = None
        for tr in SPECIES:
            if tr == te:
                continue
            a, b_ = sp == tr, sp == te
            row = {}
            for v in VAR:
                X = (np.c_[B3["dz"], B3["rz"]] if v == "nopitch"
                     else np.c_[ch3[v], B3["dz"], B3["rz"]])
                row[v] = float(bacc(y[b_], A.fit_predict(X[a], y[a], X[b_])))
            sens2[f"{tr}->{te}"] = row
    rep["sensitivity_file_duration"] = dict(
        transfer=sens2,
        means={v: float(np.mean([r[v] for r in sens2.values()])) for v in VAR})

    json.dump(rep, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)

    print("=== transfer incl. no-pitch baseline (framing A) ===")
    print(f"{'pair':<10}" + "".join(f"{v:>11}" for v in VAR))
    for p in pairs:
        print(f"{p:<10}" + "".join(f"{tab[p][v]:>11.3f}" for v in VAR))
    print(f"{'MEAN':<10}" + "".join(f"{means[v]:>11.3f}" for v in VAR))
    print("\n=== paired cluster-bootstrap Delta vs raw (95% CI) ===")
    for p in pairs:
        for v in ["allo_all", "allo_loso", "zspec"]:
            b = boot[p][f"{v}_minus_raw"]
            star = "" if (b["ci95"][0] < 0 < b["ci95"][1]) else "  <-- CI excludes 0"
            print(f"  {p:<10} {v:<10} {b['delta']:+.3f}  [{b['ci95'][0]:+.3f},{b['ci95'][1]:+.3f}]{star}")
    print("\n=== allometric residual vs species-mean centring ===")
    for s, v in eq.items():
        print(f"  {s}: obs/pred F0 = {v['observed_over_predicted_F0']:.2f}x, "
              f"corr(allo,centred)={v['corr_allo_vs_speciescentred']:.4f}, "
              f"sd(allo-centred)={v['sd_allo_minus_speciescentred_log10']:.4f} "
              f"(within-species sd logF0={v['sd_of_logF0_within_species']:.3f})")
    print("\n=== sensitivity: published pig F0 ===")
    print("  law slope", round(sens["law"]["slope"], 3), "R2", round(sens["law"]["r2"], 3),
          "pig med F0", round(sens["pig_median_f0_hz"], 1), "n", sens["n"])
    print("  means:", {k: round(v, 3) for k, v in sens["means"].items()})
    print("=== sensitivity: file duration ===")
    print("  means:", {k: round(v, 3) for k, v in rep["sensitivity_file_duration"]["means"].items()})


if __name__ == "__main__":
    main()
