"""
Stage 4 -- THE NOISE FLOOR.  Only 4 of the 6 labs can be scored for valence, so
leave-one-lab-out averages 4 folds.  Before any method difference is read as real we
need the spread of the metric under transforms that provably remove NO site information.

Null 1: random rank-k projections, many seeds  (a meaningless transform)
Null 2: bootstrap over the held-out labs of the untreated baseline
"""
import os, sys, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import site_lib as S
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import balanced_accuracy_score as B
D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "out_site")
X = np.load(os.path.join(D,"out_pigs","emb_wavlm.npy"), mmap_mode="r")[0].astype(np.float32)
meta = json.load(open(os.path.join(D,"out_pigs","meta.json")))
y = np.array([m["pol"] for m in meta]); g = np.array([m["team"] for m in meta])
labs = np.unique(g)
P = os.path.join(OUT, "nulls.json"); R = json.load(open(P)) if os.path.exists(P) else {}

def loto_preds(tf=None):
    yp = np.full_like(y, -1)
    for l in labs:
        te = g == l
        if len(np.unique(y[~te])) < 2: continue
        A, Bx = X[~te], X[te]
        if tf is not None:
            t = tf().fit(A, g[~te], y[~te]); A, Bx = t.apply(A, g[~te]), t.apply(Bx, g[te])
        a, b = S.zsc(A, Bx)
        yp[te] = LR(max_iter=3000, C=1.0, class_weight="balanced").fit(a, y[~te]).predict(b)
    return yp

# ---- Null 1: random projections that remove no site information
if "rand_null" not in R:
    t = time.time(); nul = {}
    for k in [5, 32, 128, 384]:
        vals = [float(B(y[(p := loto_preds(lambda k=k, s=s: S.RandProj(k, seed=s))) >= 0],
                        p[p >= 0])) for s in range(8)]
        nul[str(k)] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals)),
                       "min": float(np.min(vals)), "max": float(np.max(vals)), "vals": vals}
        print(f"  rand-{k:<4d} over 8 seeds: {np.mean(vals):.3f} +- {np.std(vals):.3f} "
              f"[{np.min(vals):.3f}, {np.max(vals):.3f}]", flush=True)
    R["rand_null"] = nul; json.dump(R, open(P,"w"), indent=1, default=float)
    print(f"[rand null] {time.time()-t:.0f}s")

# ---- Null 2: how much does the 4-fold average itself wobble?
if "fold_bootstrap" not in R:
    yp = loto_preds()
    scor = [l for l in labs if len(np.unique(y[g == l])) > 1]
    per = {l: float(B(y[g == l], yp[g == l])) for l in scor}
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(4000):
        pick = rng.choice(scor, size=len(scor), replace=True)
        m = np.concatenate([np.where(g == l)[0] for l in pick])
        boots.append(B(y[m], yp[m]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    R["fold_bootstrap"] = {"per_lab": per, "point": float(B(y[yp >= 0], yp[yp >= 0])),
                           "ci95": [float(lo), float(hi)], "width": float(hi - lo)}
    json.dump(R, open(P,"w"), indent=1, default=float)
    print(f"[baseline] per-lab " + ", ".join(f"{k}:{v:.2f}" for k,v in per.items()))
    print(f"  95% CI over held-out labs (cluster bootstrap): [{lo:.3f}, {hi:.3f}]  width {hi-lo:.3f}")
print("stage4 done")
