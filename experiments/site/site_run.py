"""
Can site (recording-lab) identity be erased from frozen WavLM embeddings of pig
calls, post hoc, without destroying valence?

Baseline (layer 0, n=5031, 6 labs):  valence leave-one-lab-out 0.654
                                     lab decodable at 0.941 (chance 0.167, majority 0.291)
Known negatives: INLP plateaus at 0.82; a DANN made lab decodability WORSE.

Everything below is reported as a PAIR -- valence under leave-one-lab-out, and lab
decodability from a FRESH probe that never saw the transform being fitted.
"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import site_lib as S

D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "out_site"); os.makedirs(OUT, exist_ok=True)
RES = os.path.join(OUT, "results.json")


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ data
X = np.load(os.path.join(D, "out_pigs", "emb_wavlm.npy"), mmap_mode="r")[0].astype(np.float32)
meta = json.load(open(os.path.join(D, "out_pigs", "meta.json")))
y = np.array([m["pol"] for m in meta])
g = np.array([m["team"] for m in meta])
ctx = np.array([m["ctx"] for m in meta])
labs = np.unique(g)
NL = len(labs)
CHANCE, MAJ = 1.0 / NL, max((g == l).mean() for l in labs)

R = {"setup": {"n": int(len(y)), "dim": int(X.shape[1]), "layer": 0, "labs": labs.tolist(),
               "lab_counts": {l: int((g == l).sum()) for l in labs},
               "lab_chance_balanced": CHANCE, "lab_majority_raw": float(MAJ),
               "valence_chance": 0.5}}
if os.path.exists(RES):
    try: R.update({k: v for k, v in json.load(open(RES)).items() if k != "setup"})
    except Exception: pass

def save():
    json.dump(R, open(RES, "w"), indent=1, default=float)


# ================================================================== 0. diagnostics
def diagnostics():
    d = {}
    # (i) is valence confounded with lab?
    bal = {l: float(y[g == l].mean()) for l in labs}
    d["per_lab_neg_rate"] = bal
    d["global_neg_rate"] = float(y.mean())
    # best possible valence prediction from the lab label alone
    pred = np.array([1 if bal[l] >= .5 else 0 for l in g])
    from sklearn.metrics import balanced_accuracy_score as B
    d["valence_from_lab_alone_bacc"] = float(B(y, pred))
    # contexts per lab (each context is wholly one valence)
    d["contexts_per_lab"] = {l: sorted(set(ctx[g == l].tolist())) for l in labs}
    d["n_contexts_shared_by_all_labs"] = int(len(set.intersection(
        *[set(ctx[g == l].tolist()) for l in labs])))

    # (ii) geometry of the lab mean shifts
    mu = X.mean(0)
    delta = {l: X[g == l].mean(0) - mu for l in labs}
    within = float(np.mean([X[g == l].std(0).mean() for l in labs]))
    d["mean_shift_norm"] = {l: float(np.linalg.norm(delta[l])) for l in labs}
    d["within_lab_sd_mean"] = within
    # can an UNSEEN lab's offset be predicted from the labs you have?
    cap = {}
    for l in labs:
        tr = [k for k in labs if k != l]
        mut = X[np.isin(g, tr)].mean(0)
        V = np.stack([X[g == k].mean(0) - mut for k in tr]).T        # d x 5
        Q = np.linalg.qr(V)[0]
        dl = X[g == l].mean(0) - mut
        cap[l] = float(np.linalg.norm(Q.T @ dl) / (np.linalg.norm(dl) + 1e-12))
    d["heldout_mean_shift_captured_by_train_span"] = cap
    d["heldout_mean_shift_captured_mean"] = float(np.mean(list(cap.values())))

    # (iii) second-order heterogeneity: how different are the lab covariances?
    Xz = (X - mu) / (X.std(0) + 1e-8)
    C = {l: np.cov(Xz[g == l], rowvar=False) for l in labs}
    Cp = np.cov(Xz, rowvar=False)
    d["cov_rel_frobenius_vs_pooled"] = {
        l: float(np.linalg.norm(C[l] - Cp) / np.linalg.norm(Cp)) for l in labs}
    d["per_dim_sd_ratio_max_over_min"] = float(np.median(
        np.max([Xz[g == l].std(0) for l in labs], 0) /
        np.min([Xz[g == l].std(0) for l in labs], 0)))
    # signal split: how much of the lab-discriminative power is 1st vs 2nd order
    return d


if "diagnostics" not in R:
    t = time.time(); R["diagnostics"] = diagnostics(); save()
    log(f"[diagnostics] {time.time()-t:.0f}s")
    dg = R["diagnostics"]
    log(f"  per-lab negative-valence rate: " +
        ", ".join(f"{k}={v:.2f}" for k, v in dg['per_lab_neg_rate'].items()))
    log(f"  valence predictable from lab alone: {dg['valence_from_lab_alone_bacc']:.3f}")
    log(f"  contexts shared by all 6 labs: {dg['n_contexts_shared_by_all_labs']}")
    log(f"  held-out lab's mean shift captured by the other 5 labs' span: "
        f"{dg['heldout_mean_shift_captured_mean']:.3f} "
        f"({', '.join(f'{k}={v:.2f}' for k,v in dg['heldout_mean_shift_captured_by_train_span'].items())})")
    log(f"  lab covariance vs pooled, rel. Frobenius: " +
        ", ".join(f"{k}={v:.2f}" for k, v in dg['cov_rel_frobenius_vs_pooled'].items()))
    log(f"  median per-dim SD ratio max/min across labs: {dg['per_dim_sd_ratio_max_over_min']:.2f}")


# ================================================================== 1. reference points
if "reference" not in R:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score as B
    ref = {}
    yp = np.empty_like(y)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        a, b = S.zsc(X[tr], X[te])
        yp[te] = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(a, y[tr]).predict(b)
    ref["valence_random5fold_leaky"] = float(B(y, yp))
    per = {}
    for l in labs:
        m = g == l
        if len(np.unique(y[m])) < 2: continue
        yl = y[m]; Xl = X[m]; q = np.empty_like(yl)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xl, yl):
            a, b = S.zsc(Xl[tr], Xl[te])
            q[te] = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(a, yl[tr]).predict(b)
        per[l] = float(B(yl, q))
    ref["valence_within_lab_cv"] = per
    ref["valence_within_lab_cv_mean"] = float(np.mean(list(per.values())))
    R["reference"] = ref; save()
    log(f"[reference] leaky random 5-fold valence {ref['valence_random5fold_leaky']:.3f} | "
        f"within-lab CV mean {ref['valence_within_lab_cv_mean']:.3f} "
        f"({', '.join(f'{k}:{v:.2f}' for k,v in per.items())})")


# ================================================================== 2. method sweep
METHODS = [
    ("raw",            lambda: S.Tf()),
    ("rand-5",         lambda: S.RandProj(5)),
    ("rand-32",        lambda: S.RandProj(32)),
    ("rand-128",       lambda: S.RandProj(128)),
    ("rand-384",       lambda: S.RandProj(384)),
    ("LEACE",          lambda: S.LEACE()),
    ("cLEACE",         lambda: S.LEACE(cond=True)),
    ("NAP-5",          lambda: S.NAP(5)),
    ("NAP-1",          lambda: S.NAP(1)),
    ("NAP-2",          lambda: S.NAP(2)),
    ("NAP-3",          lambda: S.NAP(3)),
    ("NAP-4",          lambda: S.NAP(4)),
    ("rand-2",         lambda: S.RandProj(2)),
    ("WCCN+NAP-5",     lambda: S.NAP(5, "wccn")),
    ("WCCN",           lambda: S.WCCN()),
    ("LEACE-T",        lambda: S.LEACE(fit_all=True)),
    ("NAP-5-T",        lambda: S.NAP(5, fit_all=True)),
    ("per-lab center", lambda: S.LabNorm("center")),
    ("per-lab scale",  lambda: S.LabNorm("scale")),
    ("per-lab z",      lambda: S.LabNorm("z")),
    ("CORAL-train",    lambda: S.CORAL("train")),
    ("CORAL-all",      lambda: S.CORAL("all")),
    ("INLP-fix-5",     lambda: S.INLP(5, "fix")),
    ("INLP-fix-32",    lambda: S.INLP(32, "fix")),
    ("INLP-orig-5",    lambda: S.INLP(5, "orig")),
    ("INLP-orig-32",   lambda: S.INLP(32, "orig")),
]
NONLIN = {"raw", "LEACE", "cLEACE", "NAP-5", "INLP-fix-32", "WCCN+NAP-5", "CORAL-train",
          "LEACE-T", "CORAL-all", "per-lab z", "rand-5"}

R.setdefault("methods", {})
log("\n" + "=" * 108)
log(f"{'method':<15} {'rk':>3} {'T':>1} | {'valence':>7} | {'lab acc':>7} {'lab bacc':>8} | "
    f"{'insample':>8} | {'quad':>6} {'MLP':>6} {'RFF':>6}")
log(f"{'':<15} {'':>3} {'':>1} | {'LOTO':>7} | {'fresh':>7} {'fresh':>8} | {'erasure':>8} | "
    f"{'':>6} {'':>6} {'':>6}")
log("=" * 108)
for name, fac in METHODS:
    e = R["methods"].get(name, {})
    t0 = time.time()
    try:
        probe = fac()
        e["rank"] = e.get("rank") or int(getattr(probe, "rank", 0))
        e.setdefault("transductive", bool(getattr(probe, "transductive", False)))
        if "valence_loto" not in e:
            e["valence_loto"] = S.valence_loto(X, y, g, fac)
        if "lab_acc" not in e:
            e["lab_acc"], e["lab_bacc"] = S.lab_linear(X, g, fac, refit_per_fold=True, y=y)
        if "lab_acc_insample" not in e:
            e["lab_acc_insample"], e["lab_bacc_insample"] = S.lab_linear(X, g, fac, refit_per_fold=False, y=y)
        if name in NONLIN:
            if "lab_quad" not in e:
                e["lab_quad"], e["lab_quad_bacc"] = S.lab_quadratic(X, g, fac, y=y)
            if "lab_mlp" not in e:
                e["lab_mlp"], e["lab_mlp_bacc"] = S.lab_mlp(X, g, fac, y=y)
            if "lab_rff" not in e:
                e["lab_rff"], e["lab_rff_bacc"] = S.lab_rff(X, g, fac, y=y)
        e["secs"] = round(time.time() - t0, 1)
        R["methods"][name] = e; save()
        f = lambda k: f"{e[k]:.3f}" if k in e else "   -  "
        log(f"{name:<15} {e['rank']:>3} {'T' if e['transductive'] else ' ':>1} | "
            f"{e['valence_loto']:>7.3f} | {e['lab_acc']:>7.3f} {e['lab_bacc']:>8.3f} | "
            f"{e['lab_acc_insample']:>8.3f} | {f('lab_quad'):>6} {f('lab_mlp'):>6} {f('lab_rff'):>6}"
            f"   [{e['secs']}s]")
    except Exception as ex:
        log(f"{name:<15} FAILED: {type(ex).__name__}: {ex}")
        R["methods"][name] = {"error": f"{type(ex).__name__}: {ex}"}; save()

log("=" * 108)
log(f"reference floors: lab chance(balanced) {CHANCE:.3f}, lab majority(raw) {MAJ:.3f}, valence chance 0.500")
log(f"wrote {RES}")
