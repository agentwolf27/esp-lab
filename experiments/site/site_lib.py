"""
Site-invariance toolkit for the pig / 6-lab confound.

Every transform is an object with .fit(X, g, y) and .apply(X, g).
Transforms that need per-lab statistics AT APPLY TIME (per-lab z-scoring, CORAL)
are flagged .transductive = True: they read unlabelled target-lab features.
Everything else is fit on training labs only and is a fixed affine map.
"""
from __future__ import annotations
import os
for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[_v] = "2"
import warnings, numpy as np
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as BACC
from sklearn.model_selection import StratifiedKFold

SEED = 0


# ---------------------------------------------------------------- linear algebra
def _sym_pow(S, p, rtol=1e-10):
    """S^p for a symmetric PSD S, pseudo-inverse style (zero out null directions)."""
    w, U = np.linalg.eigh((S + S.T) / 2.0)
    w = np.maximum(w, 0.0)
    keep = w > rtol * max(w.max(), 1e-30)
    ws = np.zeros_like(w)
    ws[keep] = w[keep] ** p
    return (U * ws) @ U.T


def onehot(g):
    labs = np.unique(g)
    return (g[:, None] == labs[None, :]).astype(np.float64), labs


def zsc(a, b):
    m, s = a.mean(0), a.std(0) + 1e-8
    return (a - m) / s, (b - m) / s


# ---------------------------------------------------------------- transforms
class Tf:
    name = "raw"; rank = 0; transductive = False; note = ""
    def fit(self, X, g, y): return self
    def apply(self, X, g): return X


class RandProj(Tf):
    """Control: project out k random orthonormal directions."""
    transductive = False
    def __init__(self, k, seed=SEED):
        self.k = k; self.rank = k; self.seed = seed; self.name = f"rand-{k}"
    def fit(self, X, g, y):
        rng = np.random.default_rng(self.seed)
        A = rng.standard_normal((X.shape[1], self.k))
        self.Q = np.linalg.qr(A)[0]                       # d x k
        return self
    def apply(self, X, g):
        return X - (X @ self.Q) @ self.Q.T


class INLP(Tf):
    """Iterative nullspace projection. mode='orig' reproduces the existing code
    (projects out the z-space coefficient vector); mode='fix' projects out the
    correct raw-space direction w/sigma."""
    def __init__(self, k, mode="fix", seed=SEED):
        self.k = k; self.rank = k; self.mode = mode; self.seed = seed
        self.name = f"INLP{'' if mode=='fix' else '-orig'}-{k}"
    def fit(self, X, g, y):
        Xw = X.astype(np.float64).copy(); dirs = []
        while len(dirs) < self.k:
            s = Xw.std(0); s = np.maximum(s, 1e-3 * s.mean() + 1e-12)
            clf = LogisticRegression(max_iter=1500, C=1.0,
                                     random_state=self.seed).fit((Xw - Xw.mean(0)) / s, g)
            W = np.atleast_2d(clf.coef_)
            grew = False
            for w in W:
                w = w / s if self.mode == "fix" else w.copy()
                for d in dirs:
                    w = w - np.dot(w, d) * d
                n = np.linalg.norm(w)
                if n < 1e-8:
                    continue
                d = w / n; dirs.append(d); Xw = Xw - np.outer(Xw @ d, d); grew = True
                if len(dirs) >= self.k:
                    break
            if not grew:
                break
        self.Q = np.array(dirs).T if dirs else np.zeros((X.shape[1], 0))
        self.rank = self.Q.shape[1]
        return self
    def apply(self, X, g):
        if self.Q.shape[1] == 0: return X
        return X - (X @ self.Q) @ self.Q.T


class LEACE(Tf):
    """LEAst-squares Concept Erasure (Belrose et al. 2023), closed form.

        r(x) = x - P (x - mu),   P = Sxx^{+1/2} Q Q^T Sxx^{-1/2},
        Q = orthonormal basis of colspace( Sxx^{-1/2} Sxz ).

    Guarantees Cov(r(X), Z) = 0 in-sample, i.e. every lab-conditional mean is
    pulled onto the global mean.  cond=True computes Sxx and Sxz POOLED WITHIN
    VALENCE STRATA, so the lab directions are estimated after valence is
    partialled out (the valence mean-difference direction is protected).
    """
    fit_all = False
    def __init__(self, cond=False, fit_all=False):
        self.cond = cond
        self.fit_all = fit_all
        self.name = ("cLEACE" if cond else "LEACE") + ("-T" if fit_all else "")
        self.note = ("transductive " if fit_all else "") + ("valence-conditional" if cond else "")
        self.transductive = fit_all
    def fit(self, X, g, y):
        X = X.astype(np.float64); n, d = X.shape
        Z, self.labs = onehot(g)
        self.mu = X.mean(0)
        if not self.cond or y is None:
            Xc, Zc = X - self.mu, Z - Z.mean(0)
            Sxx = Xc.T @ Xc / n; Sxz = Xc.T @ Zc / n
        else:
            Sxx = np.zeros((d, d)); Sxz = np.zeros((d, Z.shape[1]))
            for c in np.unique(y):
                m = y == c
                Xc = X[m] - X[m].mean(0); Zc = Z[m] - Z[m].mean(0)
                Sxx += Xc.T @ Xc / n; Sxz += Xc.T @ Zc / n
        W  = _sym_pow(Sxx, -0.5)
        Wi = _sym_pow(Sxx, +0.5)
        M = W @ Sxz
        U, s, _ = np.linalg.svd(M, full_matrices=False)
        r = int((s > 1e-8 * max(s.max(), 1e-30)).sum())
        Q = U[:, :r]
        self.P = Wi @ Q @ Q.T @ W
        self.rank = r
        return self
    def apply(self, X, g):
        return (X - (X.astype(np.float64) - self.mu) @ self.P.T).astype(np.float32)


class NAP(Tf):
    """Nuisance Attribute Projection: orthogonally project out the top-k
    eigenvectors of the BETWEEN-LAB scatter.  k=nlabs-1 removes the lab-mean
    subspace exactly.  metric='wccn' first whitens by the pooled WITHIN-lab
    covariance (classic WCCN+NAP), then unwhitens."""
    fit_all = False
    def __init__(self, k, metric="raw", fit_all=False):
        self.k = k; self.rank = k; self.metric = metric; self.fit_all = fit_all
        self.transductive = fit_all
        self.name = f"{'WCCN+' if metric=='wccn' else ''}NAP-{k}" + ("-T" if fit_all else "")
    def fit(self, X, g, y):
        X = X.astype(np.float64); n, d = X.shape
        mu = X.mean(0); self.mu = mu
        if self.metric == "wccn":
            Sw = np.zeros((d, d))
            for l in np.unique(g):
                m = g == l; A = X[m] - X[m].mean(0); Sw += A.T @ A
            Sw /= n
            self.W  = _sym_pow(Sw, -0.5); self.Wi = _sym_pow(Sw, +0.5)
        else:
            self.W = self.Wi = np.eye(d)
        Y = (X - mu) @ self.W
        Sb = np.zeros((d, d))
        for l in np.unique(g):
            m = g == l; v = Y[m].mean(0); Sb += (m.mean()) * np.outer(v, v)
        w, U = np.linalg.eigh(Sb)
        self.Q = U[:, ::-1][:, :self.k]
        return self
    def apply(self, X, g):
        Y = (X.astype(np.float64) - self.mu) @ self.W
        Y = Y - (Y @ self.Q) @ self.Q.T
        return (Y @ self.Wi + self.mu).astype(np.float32)


class WCCN(Tf):
    """Whiten by the pooled WITHIN-lab covariance.  Invertible: it cannot change
    what a linear probe can do, only what a *regularised* one does.  Included
    because it is the textbook speaker-verification answer."""
    name = "WCCN"; rank = 0
    def fit(self, X, g, y):
        X = X.astype(np.float64); n, d = X.shape
        self.mu = X.mean(0)
        Sw = np.zeros((d, d))
        for l in np.unique(g):
            m = g == l; A = X[m] - X[m].mean(0); Sw += A.T @ A
        self.W = _sym_pow(Sw / n, -0.5)
        return self
    def apply(self, X, g):
        return ((X.astype(np.float64) - self.mu) @ self.W).astype(np.float32)


class LabNorm(Tf):
    """TRANSDUCTIVE per-lab standardisation.  mode: center | scale | z."""
    transductive = True
    def __init__(self, mode="z"):
        self.mode = mode; self.name = {"center": "per-lab center", "scale": "per-lab scale",
                                       "z": "per-lab z-score"}[mode]
        self.note = "transductive"
    def fit(self, X, g, y):
        self.gm = X.mean(0); self.gs = X.std(0) + 1e-8
        return self
    def apply(self, X, g):
        Z = X.astype(np.float32).copy()
        for l in np.unique(g):
            m = g == l
            if m.sum() < 2: continue
            mu, sd = X[m].mean(0), X[m].std(0) + 1e-8
            if self.mode == "center":   Z[m] = X[m] - mu + self.gm
            elif self.mode == "scale":  Z[m] = (X[m] - mu) / sd * self.gs + mu
            else:                       Z[m] = (X[m] - mu) / sd
        return Z


class CORAL(Tf):
    """Second-order alignment: map every lab's covariance onto a common reference.
    scope='all' aligns the held-out lab too (TRANSDUCTIVE, needs unlabelled target
    audio); scope='train' aligns only the labs seen in training (inductive, but
    then the target lab is left un-aligned)."""
    def __init__(self, scope="all", shrink=1e-2):
        self.scope = scope; self.shrink = shrink
        self.transductive = (scope == "all")
        self.name = f"CORAL-{scope}"
        self.note = "transductive" if scope == "all" else "inductive"
    def fit(self, X, g, y):
        X = X.astype(np.float64); d = X.shape[1]
        self.mu = X.mean(0)
        C = np.cov(X - self.mu, rowvar=False)
        C = (1 - self.shrink) * C + self.shrink * np.trace(C) / d * np.eye(d)
        self.Cref_h = _sym_pow(C, 0.5)
        self.seen = set(np.unique(g).tolist())
        return self
    def apply(self, X, g):
        Xd = X.astype(np.float64); d = Xd.shape[1]; Z = Xd.copy()
        for l in np.unique(g):
            m = g == l
            if self.scope == "train" and l not in self.seen:  continue
            if m.sum() < d // 4:   # too few frames for a stable 768x768 covariance
                Z[m] = Xd[m] - Xd[m].mean(0) + self.mu; continue
            A = Xd[m] - Xd[m].mean(0)
            C = np.cov(A, rowvar=False)
            C = (1 - self.shrink) * C + self.shrink * np.trace(C) / d * np.eye(d)
            Z[m] = A @ _sym_pow(C, -0.5) @ self.Cref_h + self.mu
        return Z.astype(np.float32)


# ---------------------------------------------------------------- probes
def valence_loto(X, y, g, tf_factory):
    """(a) held-out-LAB valence.  The transform is re-fit on the training labs
    inside every fold; nothing about the held-out lab's LABELS is ever used."""
    yp = np.full_like(y, -1)
    for l in np.unique(g):
        te = g == l
        if len(np.unique(y[~te])) < 2:
            continue
        tf = tf_factory()
        tf = tf.fit(X, g, None) if getattr(tf, "fit_all", False) else tf.fit(X[~te], g[~te], y[~te])
        A, B = tf.apply(X[~te], g[~te]), tf.apply(X[te], g[te])
        a, b = zsc(A, B)
        yp[te] = LogisticRegression(max_iter=3000, C=1.0,
                                    class_weight="balanced").fit(a, y[~te]).predict(b)
    m = yp >= 0
    return float(BACC(y[m], yp[m]))


def _lin_probe(a, b, gtr):
    return LogisticRegression(max_iter=3000, C=1.0).fit(a, gtr).predict(b)


def lab_linear(X, g, tf_factory, refit_per_fold=True, folds=5, y=None):
    """(b) lab decodability with a FRESH probe.
    refit_per_fold=False : transform fit once on ALL data (this is the protocol the
                           earlier INLP numbers used -- generous to the eraser),
    refit_per_fold=True  : transform fit on the probe's training fold only
                           (tests whether the erasure GENERALISES)."""
    if not refit_per_fold:
        Xt = tf_factory().fit(X, g, y).apply(X, g)
    yp = np.empty(len(g), dtype=object)
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(X, g):
        if refit_per_fold:
            tf = tf_factory()
            tf = tf.fit(X, g, y) if getattr(tf, "fit_all", False) else tf.fit(X[tr], g[tr], None if y is None else y[tr])
            A, B = tf.apply(X[tr], g[tr]), tf.apply(X[te], g[te])
        else:
            A, B = Xt[tr], Xt[te]
        a, b = zsc(A, B)
        yp[te] = _lin_probe(a, b, g[tr])
    return float((yp == g).mean()), float(BACC(g, yp))


def lab_quadratic(X, g, tf_factory, folds=5, refit_per_fold=True, y=None):
    """Linear probe on [z, z**2]: the cheapest test for SECOND-ORDER lab structure
    (per-dimension variance differences) that a first-order eraser cannot touch."""
    if not refit_per_fold:
        Xt = tf_factory().fit(X, g, y).apply(X, g)
    yp = np.empty(len(g), dtype=object)
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(X, g):
        if refit_per_fold:
            tf = tf_factory()
            tf = tf.fit(X, g, y) if getattr(tf, "fit_all", False) else tf.fit(X[tr], g[tr], None if y is None else y[tr])
            A, B = tf.apply(X[tr], g[tr]), tf.apply(X[te], g[te])
        else:
            A, B = Xt[tr], Xt[te]
        a, b = zsc(A, B)
        a = np.hstack([a, a ** 2]); b = np.hstack([b, b ** 2])
        a, b = zsc(a, b)
        yp[te] = _lin_probe(a, b, g[tr])
    return float((yp == g).mean()), float(BACC(g, yp))


def lab_mlp(X, g, tf_factory, folds=5, h=256, epochs=40, seed=0, refit_per_fold=True, y=None):
    """Nonlinear lab probe (MLP).  The eraser is re-fit inside every probe fold,
    so this asks whether a nonlinear probe finds lab information that a linear
    eraser provably removed at first order."""
    import torch, torch.nn as nn
    torch.set_num_threads(2)
    if not refit_per_fold:
        Xt = tf_factory().fit(X, g, y).apply(X, g).astype(np.float32)
    labs = np.unique(g); gi = np.searchsorted(labs, g)
    yp = np.empty(len(g), dtype=int)
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(X, g):
        if refit_per_fold:
            tf = tf_factory()
            tf = tf.fit(X, g, y) if getattr(tf, "fit_all", False) else tf.fit(X[tr], g[tr], None if y is None else y[tr])
            A0, B0 = tf.apply(X[tr], g[tr]), tf.apply(X[te], g[te])
        else:
            A0, B0 = Xt[tr], Xt[te]
        torch.manual_seed(seed)
        a, b = zsc(np.asarray(A0, np.float32), np.asarray(B0, np.float32))
        A = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        B = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32))
        T = torch.from_numpy(gi[tr].astype(np.int64))
        net = nn.Sequential(nn.Linear(A.shape[1], h), nn.ReLU(), nn.Dropout(0.2),
                            nn.Linear(h, 128), nn.ReLU(), nn.Linear(128, len(labs)))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        lf = nn.CrossEntropyLoss()
        n = len(A); rng = np.random.default_rng(seed)
        for ep in range(epochs):
            idx = rng.permutation(n)
            for s0 in range(0, n, 256):
                j = idx[s0:s0 + 256]
                opt.zero_grad(); lf(net(A[j]), T[j]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            yp[te] = net(B).argmax(1).numpy()
    pred = labs[yp]
    return float((pred == g).mean()), float(BACC(g, pred))


def lab_rff(X, g, tf_factory, n_comp=1024, folds=5, seed=0, refit_per_fold=True, y=None):
    """Kernel (RBF) probe via random Fourier features -- a second nonlinear probe
    with a different inductive bias from the MLP."""
    from sklearn.kernel_approximation import RBFSampler
    if not refit_per_fold:
        Xt = tf_factory().fit(X, g, y).apply(X, g)
    yp = np.empty(len(g), dtype=object)
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(X, g):
        if refit_per_fold:
            tf = tf_factory()
            tf = tf.fit(X, g, y) if getattr(tf, "fit_all", False) else tf.fit(X[tr], g[tr], None if y is None else y[tr])
            A, B = tf.apply(X[tr], g[tr]), tf.apply(X[te], g[te])
        else:
            A, B = Xt[tr], Xt[te]
        a, b = zsc(A, B)
        rbf = RBFSampler(gamma=1.0 / a.shape[1], n_components=n_comp, random_state=seed).fit(a)
        yp[te] = _lin_probe(rbf.transform(a), rbf.transform(b), g[tr])
    return float((yp == g).mean()), float(BACC(g, yp))
