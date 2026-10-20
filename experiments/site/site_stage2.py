"""
Stage 2.

(a) IS SITE ACTUALLY THE BOTTLENECK?  Every context in this corpus is recorded by
    exactly one lab (0 contexts are shared by all 6 labs), so leave-one-lab-out is
    simultaneously a site shift AND a shift onto unseen contexts.  Leave-one-CONTEXT-out
    holds the labs fixed and changes only the context; comparing the two says how much
    of the 0.654 is site and how much is context novelty.

(b) NONLINEAR ERASURE.  LEACE only equalises FIRST-order (mean) statistics.  Erase in a
    random-Fourier-feature space instead: zeroing the cross-covariance there kills every
    function in the induced RKHS ball, i.e. a whole class of nonlinear lab predictors.
    Then ask the same two questions: is lab gone (linear AND MLP probe), does valence survive?
"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import site_lib as S
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as B

D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "out_site"); os.makedirs(OUT, exist_ok=True)
RES2 = os.path.join(OUT, "stage2.json")
X = np.load(os.path.join(D, "out_pigs", "emb_wavlm.npy"), mmap_mode="r")[0].astype(np.float32)
meta = json.load(open(os.path.join(D, "out_pigs", "meta.json")))
y = np.array([m["pol"] for m in meta]); g = np.array([m["team"] for m in meta])
ctx = np.array([m["ctx"] for m in meta])
R = json.load(open(RES2)) if os.path.exists(RES2) else {}
def save(): json.dump(R, open(RES2, "w"), indent=1, default=float)
def log(*a): print(*a, flush=True)


def grouped_cv(X, y, grp, tf_factory=None, min_per_side=8):
    """Leave-one-group-out with an optional transform re-fitted on the training side."""
    yp = np.full_like(y, -1)
    for u in np.unique(grp):
        te = grp == u
        if len(np.unique(y[~te])) < 2 or te.sum() < min_per_side:
            continue
        A, Bx = X[~te], X[te]
        if tf_factory is not None:
            tf = tf_factory()
            tf = tf.fit(X, grp, None) if getattr(tf, "fit_all", False) else tf.fit(A, grp[~te], y[~te])
            A, Bx = tf.apply(A, grp[~te]), tf.apply(Bx, grp[te])
        a, b = S.zsc(A, Bx)
        yp[te] = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(a, y[~te]).predict(b)
    m = yp >= 0
    if m.sum() == 0:
        return float('nan'), 0
    return float(B(y[m], yp[m])), int(m.sum())


# --------------------------------------------------------------- (a) site vs context
if "shift_decomposition" not in R:
    t = time.time(); d = {}
    d["LOTO_lab"]      = grouped_cv(X, y, g)                       # unseen lab + unseen contexts
    d["LOCO_context"]  = grouped_cv(X, y, ctx)                     # unseen context, labs all seen
    # unseen context WITHIN a single lab: no site shift at all
    per = {}
    for l in np.unique(g):
        m = g == l
        if len(np.unique(y[m])) < 2 or len(np.unique(ctx[m])) < 2: continue
        v, nsc = grouped_cv(X[m], y[m], ctx[m])
        if nsc: per[l] = v
    d["LOCO_within_lab"] = per
    d["LOCO_within_lab_mean"] = float(np.mean(list(per.values()))) if per else None
    # random split (leaky both ways) for the ceiling
    R["shift_decomposition"] = d; save()
    log(f"[shift decomposition] {time.time()-t:.0f}s")
    log(f"  leave-one-LAB-out      : {d['LOTO_lab'][0]:.3f}  (n scored {d['LOTO_lab'][1]})   <- site shift + unseen contexts")
    log(f"  leave-one-CONTEXT-out  : {d['LOCO_context'][0]:.3f}  (n scored {d['LOCO_context'][1]})   <- unseen context only, all labs in train")
    log(f"  leave-one-context-out, WITHIN one lab: mean {d['LOCO_within_lab_mean']:.3f} "
        f"({', '.join(f'{k}:{v:.2f}' for k,v in per.items())})   <- no site shift at all")


# --------------------------------------------------------------- (b) RFF-space erasure
class RFF:
    """z-score -> random Fourier features -> (optional) LEACE inside the feature space."""
    fit_all = False
    def __init__(self, n_comp=1024, erase=True, fit_all=False, seed=0):
        self.n = n_comp; self.erase = erase; self.fit_all = fit_all; self.seed = seed
        self.transductive = fit_all
        self.rank = 5 if erase else 0
        self.name = f"RFF{self.n}" + ("+LEACE" if erase else "") + ("-T" if fit_all else "")
    def _phi(self, X):
        Z = (X - self.m) / self.s
        return np.sqrt(2.0 / self.n) * np.cos(Z @ self.W + self.b)
    def fit(self, X, g, y):
        rng = np.random.default_rng(self.seed)
        self.m, self.s = X.mean(0), X.std(0) + 1e-8
        d = X.shape[1]
        self.W = rng.standard_normal((d, self.n)) * np.sqrt(2.0 / d)   # gamma = 1/d
        self.b = rng.uniform(0, 2 * np.pi, self.n)
        self.er = S.LEACE().fit(self._phi(X), g, y) if self.erase else None
        return self
    def apply(self, X, g):
        P = self._phi(X)
        return self.er.apply(P, g).astype(np.float32) if self.er is not None else P.astype(np.float32)


if "rff" not in R:
    d = {}
    for tag, fac in [("RFF (no erasure)", lambda: RFF(erase=False)),
                     ("RFF + LEACE",      lambda: RFF(erase=True)),
                     ("RFF + LEACE-T",    lambda: RFF(erase=True, fit_all=True))]:
        t = time.time(); e = {}
        e["valence_loto"] = S.valence_loto(X, y, g, fac)
        e["lab_acc"], e["lab_bacc"] = S.lab_linear(X, g, fac, refit_per_fold=True)
        e["lab_mlp"], e["lab_mlp_bacc"] = S.lab_mlp(X, g, fac, epochs=25)
        e["secs"] = round(time.time() - t, 1); d[tag] = e
        log(f"  {tag:<18} valence {e['valence_loto']:.3f} | lab linear {e['lab_acc']:.3f} "
            f"| lab MLP {e['lab_mlp']:.3f}   [{e['secs']}s]")
        R["rff"] = d; save()
save()
log(f"wrote {RES2}")


# --------------------------------------------------------------- (c) why erasure costs the task
if "entanglement" not in R:
    from sklearn.linear_model import LogisticRegression as LR
    t = time.time(); d = {}
    mu = X.mean(0)
    V = np.stack([X[g == l].mean(0) - mu for l in np.unique(g)]).T      # d x 6, rank 5
    Q = np.linalg.qr(V)[0][:, :len(np.unique(g)) - 1]                   # the lab-mean subspace
    d["lab_subspace_rank"] = int(Q.shape[1])
    # where does the VALENCE direction live?
    Xz = (X - mu) / (X.std(0) + 1e-8)
    w = LR(max_iter=3000, C=1.0, class_weight="balanced").fit(Xz, y).coef_[0]
    w = (w / (X.std(0) + 1e-8)); w /= np.linalg.norm(w)                 # back to raw space
    d["valence_dir_in_lab_subspace"] = float(np.linalg.norm(Q.T @ w))
    d["random_dir_in_lab_subspace_expected"] = float(np.sqrt(Q.shape[1] / X.shape[1]))
    d["overlap_vs_random"] = d["valence_dir_in_lab_subspace"] / d["random_dir_in_lab_subspace_expected"]
    # split the representation and score valence on each half
    Xin  = (X - mu) @ Q                       # 5 dims: ONLY the site subspace
    Xout = (X - mu) - ((X - mu) @ Q) @ Q.T    # its orthogonal complement
    d["valence_loto_site_subspace_only"] = grouped_cv(Xin.astype(np.float32), y, g)[0]
    d["valence_loto_complement"]         = grouped_cv(Xout.astype(np.float32), y, g)[0]
    d["valence_loto_full"]               = grouped_cv(X, y, g)[0]
    R["entanglement"] = d; save()
    log(f"[entanglement] {time.time()-t:.0f}s")
    log(f"  valence direction's norm inside the {d['lab_subspace_rank']}-d site subspace: "
        f"{d['valence_dir_in_lab_subspace']:.3f}  (a random direction: "
        f"{d['random_dir_in_lab_subspace_expected']:.3f}; ratio {d['overlap_vs_random']:.1f}x)")
    log(f"  valence LOTO from the {d['lab_subspace_rank']} site dimensions ALONE : {d['valence_loto_site_subspace_only']:.3f}")
    log(f"  valence LOTO from the 763-d complement          : {d['valence_loto_complement']:.3f}")
    log(f"  valence LOTO from the full 768-d embedding      : {d['valence_loto_full']:.3f}")
save()
log("stage2 complete")
