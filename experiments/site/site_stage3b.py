"""Is the transductive site-subspace projection (0.718) real, or one lucky fold?"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import site_lib as S
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import balanced_accuracy_score as B
D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "out_site")
X = np.load(os.path.join(D,"out_pigs","emb_wavlm.npy"), mmap_mode="r")[0].astype(np.float32)
meta = json.load(open(os.path.join(D,"out_pigs","meta.json")))
y = np.array([m["pol"] for m in meta]); g = np.array([m["team"] for m in meta])
labs = np.unique(g); P = os.path.join(OUT,"stage2.json"); R = json.load(open(P))

def per_fold(proj):
    out = {}; yp = np.full_like(y, -1)
    for l in labs:
        te = g == l
        if len(np.unique(y[~te])) < 2 or len(np.unique(y[te])) < 2: continue
        A, Bx = proj(X[~te], g[~te], X[te], g[te])
        a, b = S.zsc(A, Bx)
        p = LR(max_iter=3000, C=1.0, class_weight="balanced").fit(a, y[~te]).predict(b)
        yp[te] = p; out[l] = float(B(y[te], p))
    m = yp >= 0
    return float(B(y[m], yp[m])), out

def siteQ(Xa, ga):
    mu = Xa.mean(0)
    V = np.stack([Xa[ga==l].mean(0)-mu for l in np.unique(ga)]).T
    return mu, np.linalg.qr(V)[0][:, :len(np.unique(ga))-1]

d = {}
d["full"], d["full_per_lab"] = per_fold(lambda a,b,c,e: (a, c))
muT, QT = siteQ(X, g)
d["site_T"], d["site_T_per_lab"] = per_fold(lambda a,b,c,e: ((a-muT)@QT, (c-muT)@QT))
# matched 5-d RANDOM subspace, 10 seeds
acc = []
for sd in range(10):
    Qr = np.linalg.qr(np.random.default_rng(sd).standard_normal((X.shape[1], QT.shape[1])))[0]
    acc.append(per_fold(lambda a,b,c,e,Q=Qr: (a@Q, c@Q))[0])
d["rand5_mean"], d["rand5_sd"], d["rand5_max"] = float(np.mean(acc)), float(np.std(acc)), float(np.max(acc))
# control: subspace from 5 lab means but with the HELD-OUT lab's mean replaced by a random lab's
# (isolates "does the target lab's own mean matter?")
d["site_T_dim"] = int(QT.shape[1])
R["site_subspace_probe"] = d
json.dump(R, open(P,"w"), indent=1, default=float)
print(f"full 768-d            : {d['full']:.3f}   per-lab " + ", ".join(f"{k}:{v:.2f}" for k,v in d['full_per_lab'].items()))
print(f"site subspace ({d['site_T_dim']}-d, T): {d['site_T']:.3f}   per-lab " + ", ".join(f"{k}:{v:.2f}" for k,v in d['site_T_per_lab'].items()))
print(f"random {d['site_T_dim']}-d control  : {d['rand5_mean']:.3f} +- {d['rand5_sd']:.3f}  (best of 10 seeds {d['rand5_max']:.3f})")
