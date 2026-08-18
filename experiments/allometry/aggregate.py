"""Aggregate tests over the 6 directed pairs + why-it-fails coefficients."""
from __future__ import annotations
import json, os, warnings
import numpy as np
warnings.filterwarnings("ignore")
from scipy import stats
from sklearn.linear_model import LogisticRegression
import analyze as A

OUT = os.path.dirname(os.path.abspath(__file__))
rep = json.load(open(os.path.join(OUT, "results.json")))
tab = rep["transfer_with_nopitch_baseline"]
pairs = list(tab.keys())
VAR = ["nopitch", "raw", "allo_all", "allo_loso", "zspec", "center"]

agg = {}
for v in VAR:
    if v == "raw":
        continue
    for base in ["raw", "zspec", "nopitch"]:
        if v == base:
            continue
        dl = np.array([tab[p][v] - tab[p][base] for p in pairs])
        try:
            w = float(stats.wilcoxon(dl).pvalue)
        except Exception:
            w = np.nan
        agg[f"{v}_vs_{base}"] = dict(
            mean_delta=float(dl.mean()), median_delta=float(np.median(dl)),
            n_pairs_better=int((dl > 0).sum()), n_pairs=len(dl),
            sign_test_p=float(stats.binomtest((dl > 0).sum(), len(dl)).pvalue),
            wilcoxon_p=w, per_pair=dict(zip(pairs, dl.round(4).tolist())))
rep["aggregate_tests_over_6_pairs"] = agg

# ---- why does it fail? direction each species' own probe assigns to each cue
d = A.load()
d = d[np.isfinite(d.log_f0) & np.isfinite(d.log_dur) & np.isfinite(d.log_rms)].copy()
sp, y = d.species.values, d.pol.values
ch, law, _ = A.pitch_channels(d, "cat")
why = {}
for s in A.SPECIES:
    k = sp == s
    X = np.c_[ch["zspec"][k], A.zwithin(d.log_dur.values, sp)[k], A.zwithin(d.log_rms.values, sp)[k]]
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    c = LogisticRegression(max_iter=3000, class_weight="balanced").fit(X, y[k]).coef_[0]
    why[s] = dict(pitch=float(c[0]), dur_call=float(c[1]), energy=float(c[2]))
rep["why_probe_coefficients_negative_is_class1"] = why
cs = {}
for a in A.SPECIES:
    for b in A.SPECIES:
        if a < b:
            va = np.array([why[a][k] for k in ("pitch", "dur_call", "energy")])
            vb = np.array([why[b][k] for k in ("pitch", "dur_call", "energy")])
            cs[f"{a}~{b}"] = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
rep["affect_axis_cosine_between_species"] = cs

json.dump(rep, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)
print("=== aggregate over 6 directed pairs ===")
for k, v in agg.items():
    print(f"  {k:<24} mean_delta={v['mean_delta']:+.4f}  better in {v['n_pairs_better']}/6  "
          f"sign p={v['sign_test_p']:.3f}  wilcoxon p={v['wilcoxon_p']:.3f}")
print("\n=== each species' own affect probe (standardised coefs, +ve => predicts NEGATIVE affect) ===")
for s, v in why.items():
    print(f"  {s:4s} pitch={v['pitch']:+.3f}  dur_call={v['dur_call']:+.3f}  energy={v['energy']:+.3f}")
print("  cosine between species' axes:", {k: round(v, 3) for k, v in cs.items()})
