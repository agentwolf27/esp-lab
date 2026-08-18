"""
Stage 5 -- PAIRED comparisons.  The untreated baseline's own 95% CI is ~0.20 wide
(4 scorable labs), so comparing marginal numbers is hopeless.  But every method is
evaluated on the SAME folds and the SAME calls, so the paired difference is far better
determined.  Per-lab deltas + a paired cluster bootstrap over held-out labs.
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
labs = np.unique(g); P = os.path.join(OUT,"nulls.json"); R = json.load(open(P))
SCOR = [l for l in labs if len(np.unique(y[g == l])) > 1]

def preds(fac):
    yp = np.full_like(y, -1)
    for l in labs:
        te = g == l
        if len(np.unique(y[~te])) < 2: continue
        tf = fac()
        tf = tf.fit(X, g, None) if getattr(tf, "fit_all", False) else tf.fit(X[~te], g[~te], y[~te])
        A, Bx = tf.apply(X[~te], g[~te]), tf.apply(X[te], g[te])
        a, b = S.zsc(A, Bx)
        yp[te] = LR(max_iter=3000, C=1.0, class_weight="balanced").fit(a, y[~te]).predict(b)
    return yp

CASES = [("raw", lambda: S.Tf()), ("rand-384", lambda: S.RandProj(384)),
         ("LEACE", lambda: S.LEACE()), ("NAP-2", lambda: S.NAP(2)), ("NAP-5", lambda: S.NAP(5)),
         ("LEACE-T", lambda: S.LEACE(fit_all=True)), ("CORAL-all", lambda: S.CORAL("all")),
         ("per-lab z", lambda: S.LabNorm("z"))]
Pr = {}
for n, f in CASES:
    t = time.time(); Pr[n] = preds(f); print(f"  {n} preds [{time.time()-t:.0f}s]", flush=True)

base = Pr["raw"]; rng = np.random.default_rng(0); out = {}
idx = {l: np.where(g == l)[0] for l in SCOR}
for n in Pr:
    per = {l: float(B(y[idx[l]], Pr[n][idx[l]]) - B(y[idx[l]], base[idx[l]])) for l in SCOR}
    ds = []
    for _ in range(4000):
        pick = rng.choice(SCOR, size=len(SCOR), replace=True)
        m = np.concatenate([idx[l] for l in pick])
        ds.append(B(y[m], Pr[n][m]) - B(y[m], base[m]))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    out[n] = {"point": float(B(y[np.concatenate([idx[l] for l in SCOR])],
                               Pr[n][np.concatenate([idx[l] for l in SCOR])])),
              "delta": float(np.mean(ds)), "ci95": [float(lo), float(hi)],
              "per_lab_delta": per,
              "sig": bool(lo > 0 or hi < 0)}
R["paired"] = out; R["scorable_labs"] = SCOR
json.dump(R, open(P, "w"), indent=1, default=float)
print(f"\n{'method':<12} {'valence':>7} {'Δ vs raw':>9}  {'95% CI (paired)':>18}  sig   per-lab Δ")
for n, e in out.items():
    print(f"{n:<12} {e['point']:>7.3f} {e['delta']:>+9.3f}  [{e['ci95'][0]:+.3f},{e['ci95'][1]:+.3f}]"
          f"   {'YES' if e['sig'] else ' no'}   " +
          ", ".join(f"{k}:{v:+.2f}" for k, v in e['per_lab_delta'].items()))
