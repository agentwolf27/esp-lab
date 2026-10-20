"""
Stage 3.  Stage 2 turned up the decisive number: the 5-dimensional SITE-MEAN subspace,
on its own, predicts valence across labs BETTER than the full 768-d embedding.  That
inverts the premise of the whole erasure programme, so it needs to be done carefully:
the subspace in stage 2 was estimated using the held-out lab's own mean (label-free, but
transductive).  Here we redo it fold-honestly, with the subspace estimated from the
TRAINING labs only, plus a random-subspace control at matched dimension.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, site_lib as S
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import balanced_accuracy_score as B

D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "out_site")
X = np.load(os.path.join(D, "out_pigs", "emb_wavlm.npy"), mmap_mode="r")[0].astype(np.float32)
meta = json.load(open(os.path.join(D, "out_pigs", "meta.json")))
y = np.array([m["pol"] for m in meta]); g = np.array([m["team"] for m in meta])
P = os.path.join(OUT, "stage2.json"); R = json.load(open(P))
labs = np.unique(g)


def loto_on(fn, seed=0):
    """fn(Xtr, gtr, Xte, gte) -> (Atr, Ate). Everything re-derived inside the fold."""
    yp = np.full_like(y, -1)
    for l in labs:
        te = g == l
        if len(np.unique(y[~te])) < 2: continue
        A, Bx = fn(X[~te], g[~te], X[te], g[te])
        a, b = S.zsc(A, Bx)
        yp[te] = LR(max_iter=3000, C=1.0, class_weight="balanced").fit(a, y[~te]).predict(b)
    m = yp >= 0
    return float(B(y[m], yp[m]))


def site_basis(Xtr, gtr):
    mu = Xtr.mean(0)
    V = np.stack([Xtr[gtr == l].mean(0) - mu for l in np.unique(gtr)]).T
    Q = np.linalg.qr(V)[0][:, :len(np.unique(gtr)) - 1]
    return mu, Q


d = R.get("subspace_decomposition", {})
t = time.time()
# --- inductive: site subspace estimated from the TRAINING labs only
def f_in(Xtr, gtr, Xte, gte):
    mu, Q = site_basis(Xtr, gtr)
    return (Xtr - mu) @ Q, (Xte - mu) @ Q
d["site_subspace_only_inductive"] = loto_on(f_in)
d["site_subspace_dim_inductive"] = int(len(labs) - 2)      # 5 train labs -> rank 4

def f_out(Xtr, gtr, Xte, gte):
    mu, Q = site_basis(Xtr, gtr)
    return (Xtr - mu) - ((Xtr - mu) @ Q) @ Q.T, (Xte - mu) - ((Xte - mu) @ Q) @ Q.T
d["complement_inductive"] = loto_on(f_out)

# --- matched-dimension RANDOM subspace control (same #dims, no site information)
accs = []
for sd in range(5):
    rng = np.random.default_rng(sd)
    Qr = np.linalg.qr(rng.standard_normal((X.shape[1], len(labs) - 2)))[0]
    accs.append(loto_on(lambda a, b, c, e, Qr=Qr: (a @ Qr, c @ Qr)))
d["random_subspace_only_matched_dim"] = float(np.mean(accs))
d["random_subspace_only_sd"] = float(np.std(accs))

# --- transductive versions (subspace uses the held-out lab's own mean; label-free)
def f_in_T(Xtr, gtr, Xte, gte):
    mu, Q = site_basis(X, g)
    return (Xtr - mu) @ Q, (Xte - mu) @ Q
d["site_subspace_only_transductive"] = loto_on(f_in_T)
d["site_subspace_dim_transductive"] = int(len(labs) - 1)

# --- and the trivial upper reference: the lab-mean vector itself as the only feature
d["full"] = loto_on(lambda a, b, c, e: (a, c))
R["subspace_decomposition"] = d
json.dump(R, open(P, "w"), indent=1, default=float)
print(f"[subspace decomposition] {time.time()-t:.0f}s")
print(f"  full 768-d embedding                                  : {d['full']:.3f}")
print(f"  site-mean subspace only, INDUCTIVE ({d['site_subspace_dim_inductive']}-d, train labs)   : {d['site_subspace_only_inductive']:.3f}")
print(f"  site-mean subspace only, transductive ({d['site_subspace_dim_transductive']}-d, all labs): {d['site_subspace_only_transductive']:.3f}")
print(f"  RANDOM subspace of the same {d['site_subspace_dim_inductive']} dims (control)        : {d['random_subspace_only_matched_dim']:.3f} +- {d['random_subspace_only_sd']:.3f}")
print(f"  orthogonal complement, INDUCTIVE ({X.shape[1]-d['site_subspace_dim_inductive']}-d)         : {d['complement_inductive']:.3f}")
