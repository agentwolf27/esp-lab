"""Paired tests between conditions, at each condition's own best layer.

Same folds (StratifiedKFold(5, shuffle=True, random_state=0)) for every
condition, so predictions are paired per call -> exact McNemar + paired
bootstrap CI on the accuracy difference.
"""
import torch
torch.set_num_threads(2)

import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import binomtest

D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "out_multiband")
meta = json.load(open(os.path.join(OUT, "meta.json")))
res = json.load(open(os.path.join(OUT, "results.json")))
y = np.array(meta["labels"])
NB = 4


def preds(X, y):
    yp = np.empty_like(y)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        sc = StandardScaler().fit(X[tr])
        yp[te] = LogisticRegression(max_iter=3000, C=0.5).fit(
            sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    return yp


def feats(enc, cond, li):
    load = lambda c: np.load(os.path.join(OUT, f"emb_{enc}_{c}.npy"), mmap_mode="r")
    if cond == "multiband_mean":
        return np.mean([np.asarray(load(f"band{k}")[li], np.float32) for k in range(NB)], 0)
    if cond == "multiband_concat":
        return np.concatenate([np.asarray(load(f"band{k}")[li], np.float32) for k in range(NB)], 1)
    if cond.startswith("multiband_band"):
        return np.asarray(load(f"band{cond[14]}")[li], np.float32)
    return np.asarray(load(cond)[li], np.float32)


PAIRS = [("timeexp4x", "multiband_concat"),
         ("timeexp4x", "multiband_mean"),
         ("multiband_concat", "multiband_band0_only"),
         ("multiband_mean", "multiband_band0_only"),
         ("multiband_concat", "baseband"),
         ("baseband", "multiband_band0_only")]

out = {}
rng = np.random.default_rng(0)
for enc in [e for e in ["wavlm", "aves-bio"] if f"{e}_baseband" in res]:
    conds = sorted({c for p in PAIRS for c in p})
    P = {}
    for c in conds:
        li = res[f"{enc}_{c}"]["best_layer"]
        P[c] = preds(feats(enc, c, li), y)
        got, want = (P[c] == y).mean(), res[f"{enc}_{c}"]["best"]
        if abs(got - want) > 1e-9:
            print(f"  WARNING {enc}/{c}: reproduced {got:.4f} != results.json {want:.4f}")
    print(f"\n=== {enc} (n={len(y)}) ===")
    for a, b in PAIRS:
        ca, cb = P[a] == y, P[b] == y
        n01 = int((~ca & cb).sum()); n10 = int((ca & ~cb).sum())
        p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        d = float(ca.mean() - cb.mean())
        bs = np.array([(ca[i] .mean() - cb[i].mean())
                       for i in rng.integers(0, len(y), (2000, len(y)))])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[f"{enc}|{a}|vs|{b}"] = {"acc_a": float(ca.mean()), "acc_b": float(cb.mean()),
                                    "diff": d, "ci95": [float(lo), float(hi)],
                                    "mcnemar_b_only": n01, "mcnemar_a_only": n10,
                                    "mcnemar_p": float(p)}
        star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
        print(f"  {a:<20s} - {b:<22s} {d:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  "
              f"McNemar p={p:.2g} {star}")

json.dump(out, open(os.path.join(OUT, "paired_tests.json"), "w"), indent=1)
print("\nwrote paired_tests.json")
