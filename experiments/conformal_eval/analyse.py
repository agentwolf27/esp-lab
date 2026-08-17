"""Stage 2: split conformal on real bioacoustic embeddings, marginal vs
per-individual / per-lab coverage.

Uses the project's own conformal module (conformal_pam/src/conformal.py), which
until now had only ever been run on synthetic data.

Set-valued binary predictor
---------------------------
For each clip we form a 2-column score matrix S = [P(positive), P(negative)]
and a one-hot label matrix Y.  A label is included in the prediction set iff its
score clears the conformal threshold, so the set is a subset of
{positive, negative}: size 1 = a committed call, size 2 = "I don't know",
size 0 = "neither label is plausible".  With one-hot Y the module's
`split_conformal_threshold` is exactly the LAC / THR conformal classifier and
`coverage` is exactly P(true label in set).

Designs
-------
A  group-disjoint  : calibrate on some individuals/labs, test on held-out ones.
                     Deployment-realistic; exchangeability is BROKEN on purpose.
B  within-group    : random 50/50 split inside every group.  Exchangeable
                     control (task 6) and the only regime in which
                     group-conditional Mondrian has any calibration data.
Both are repeated over many splits and pooled, so nothing rests on one draw.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import itertools
import json
import sys

import numpy as np

CTX = "/private/tmp/claude-501/-Volumes-SSD/eb358682-62a6-4aab-8b76-0d59bcd47add/scratchpad/ctx"
OUT = os.path.join(CTX, "out_conformal")
sys.path.insert(0, "/Users/vish/esp-lab/conformal_pam/src")

from conformal import (                                    # noqa: E402
    conformal_quantile,
    conformal_risk_control,
    coverage,
    false_negative_rate,
    mean_set_size,
    mondrian_thresholds,
    predict_sets,
    split_conformal_threshold,
)

ALPHA = 0.10
NOMINAL = 1 - ALPHA
LAYERS = (9, 6)
R_SPLITS = 40
MIN_GROUP_CAL = 10        # ceil((n+1)*0.9)/n <= 1 needs n >= 9 at alpha=0.10
DATASETS = ("cats", "dogs", "pigs")


# ---------------------------------------------------------------- helpers
def build(p1, y):
    S = np.column_stack([1.0 - p1, p1])
    Y = np.zeros(S.shape, dtype=bool)
    Y[np.arange(len(y)), y] = True
    return S, Y


def feasible(pos, alpha=ALPHA):
    """Can this many calibration points support a 1-alpha guarantee at all?"""
    return pos.size >= MIN_GROUP_CAL and np.isfinite(conformal_quantile(1.0 - pos, alpha))


def th_predclass(Scal, Ycal, Ste, alpha=ALPHA, min_count=20):
    """Mondrian with the taxonomy = the model's PREDICTED class (a function of x
    only, so still a valid Mondrian taxonomy)."""
    pooled = split_conformal_threshold(Scal, Ycal, alpha)
    out = np.full(len(Ste), pooled, dtype=float)
    pc, pt = Scal.argmax(1), Ste.argmax(1)
    for c in (0, 1):
        m = pc == c
        if m.sum() < min_count or not feasible(Scal[m][Ycal[m]], alpha):
            continue
        out[pt == c] = split_conformal_threshold(Scal[m], Ycal[m], alpha)
    return out[:, None]


def th_group(Scal, Ycal, gcal, gte, pooled, alpha=ALPHA):
    """Mondrian conditioned on GROUP; pooled threshold as fallback for groups
    with no / too little calibration data."""
    out = np.full(len(gte), pooled, dtype=float)
    own = []
    for grp in np.unique(gte):
        m = gcal == grp
        if m.sum() == 0:
            continue
        if not feasible(Scal[m][Ycal[m]], alpha):
            continue
        out[gte == grp] = split_conformal_threshold(Scal[m], Ycal[m], alpha)
        own.append(str(grp))
    return out[:, None], own


class Acc:
    """Accumulates per-clip covered / set-size over repeated splits."""

    def __init__(self, n):
        self.cov = np.zeros(n); self.size = np.zeros(n)
        self.s2 = np.zeros(n); self.s0 = np.zeros(n); self.cnt = np.zeros(n)
        self.own = []

    def add(self, te_idx, S, Y, y, T):
        sets = predict_sets(S[te_idx], T)
        cov = sets[np.arange(len(te_idx)), y[te_idx]]
        sz = sets.sum(1)
        self.cov[te_idx] += cov
        self.size[te_idx] += sz
        self.s2[te_idx] += (sz == 2)
        self.s0[te_idx] += (sz == 0)
        self.cnt[te_idx] += 1


def summarize(acc, y, g, ctx):
    ok = acc.cnt > 0
    tot = acc.cnt[ok].sum()
    per = {}
    for grp in sorted(np.unique(g)):
        m = (g == grp) & ok
        if not m.any():
            continue
        c = acc.cnt[m].sum()
        per[str(grp)] = {
            "n": int((g == grp).sum()),
            "n_pos": int(((g == grp) & (y == 0)).sum()),
            "n_neg": int(((g == grp) & (y == 1)).sum()),
            "coverage": float(acc.cov[m].sum() / c),
            "set_size": float(acc.size[m].sum() / c),
            "frac_size2": float(acc.s2[m].sum() / c),
            "frac_size0": float(acc.s0[m].sum() / c),
        }
    cov = np.array([v["coverage"] for v in per.values()])
    sz = np.array([v["set_size"] for v in per.values()])
    big = np.array([v["n"] >= 10 for v in per.values()])

    def spread(a):
        q1, q2, q3 = np.percentile(a, [25, 50, 75])
        return {"min": float(a.min()), "q25": float(q1), "median": float(q2),
                "q75": float(q3), "max": float(a.max()), "iqr": float(q3 - q1)}

    own_names = sorted({n for lst in acc.own for n in lst}) if acc.own else []
    return {
        "marginal_coverage": float(acc.cov[ok].sum() / tot),
        "marginal_set_size": float(acc.size[ok].sum() / tot),
        "marginal_frac_size2": float(acc.s2[ok].sum() / tot),
        "marginal_frac_size0": float(acc.s0[ok].sum() / tot),
        "n_groups": len(per),
        "coverage_spread": spread(cov),
        "coverage_spread_groups_n>=10": spread(cov[big]) if big.any() else None,
        "n_groups_n>=10": int(big.sum()),
        "n_groups_below_0.80_n>=10": int((cov[big] < 0.80).sum()) if big.any() else None,
        "set_size_spread": spread(sz),
        "n_groups_below_0.80": int((cov < 0.80).sum()),
        "n_groups_below_nominal": int((cov < NOMINAL).sum()),
        "groups_below_0.80": sorted(
            [k for k, v in per.items() if v["coverage"] < 0.80],
            key=lambda k: per[k]["coverage"]),
        "worst_group": min(per, key=lambda k: per[k]["coverage"]),
        "worst_group_coverage": float(cov.min()),
        "mean_groups_with_own_threshold": (
            float(np.mean([len(l) for l in acc.own])) if acc.own else 0.0),
        "groups_with_own_threshold": own_names,
        "per_group": per,
    }


# ---------------------------------------------------------------- designs
def design_A(S, Y, y, g, rng, r_splits=R_SPLITS):
    """Group-disjoint calibration / test."""
    groups = np.unique(g)
    k = len(groups) // 2
    if len(groups) <= 8:                       # pigs: enumerate every partition
        parts = [np.array(c) for c in itertools.combinations(groups, k)]
    else:
        parts, seen = [], set()
        while len(parts) < r_splits:
            c = tuple(sorted(rng.choice(groups, k, replace=False)))
            if c in seen:
                continue
            seen.add(c); parts.append(np.array(c))

    accs = {n: Acc(len(y)) for n in
            ("pooled", "mondrian_class", "mondrian_predclass", "mondrian_group")}
    n_used = 0
    for cal_g in parts:
        cal = np.isin(g, cal_g)
        te = ~cal
        if len(np.unique(y[cal])) < 2 or not te.any():
            continue
        n_used += 1
        ti = np.where(te)[0]
        pooled = split_conformal_threshold(S[cal], Y[cal], ALPHA)
        accs["pooled"].add(ti, S, Y, y, pooled)
        accs["mondrian_class"].add(
            ti, S, Y, y, mondrian_thresholds(S[cal], Y[cal], ALPHA, min_count=20))
        accs["mondrian_predclass"].add(
            ti, S, Y, y, th_predclass(S[cal], Y[cal], S[te]))
        Tg, n_own = th_group(S[cal], Y[cal], g[cal], g[te], pooled)
        accs["mondrian_group"].add(ti, S, Y, y, Tg)
        accs["mondrian_group"].own.append(n_own)
    return accs, n_used


def design_B(S, Y, y, g, rng, r_splits=R_SPLITS):
    """Random 50/50 split INSIDE every group: exchangeable control, and the only
    regime where a group-conditional threshold exists for the test group."""
    accs = {n: Acc(len(y)) for n in ("pooled", "mondrian_class", "mondrian_group")}
    for _ in range(r_splits):
        cal = np.zeros(len(y), bool)
        for grp in np.unique(g):
            idx = np.where(g == grp)[0]
            cal[rng.permutation(idx)[: len(idx) // 2]] = True
        te = ~cal
        if len(np.unique(y[cal])) < 2 or not te.any():
            continue
        ti = np.where(te)[0]
        pooled = split_conformal_threshold(S[cal], Y[cal], ALPHA)
        accs["pooled"].add(ti, S, Y, y, pooled)
        accs["mondrian_class"].add(
            ti, S, Y, y, mondrian_thresholds(S[cal], Y[cal], ALPHA, min_count=20))
        Tg, n_own = th_group(S[cal], Y[cal], g[cal], g[te], pooled)
        accs["mondrian_group"].add(ti, S, Y, y, Tg)
        accs["mondrian_group"].own.append(n_own)
    return accs


# ---------------------------------------------------------------- sanity
def sanity_synthetic(n=400, trials=2000, alpha=ALPHA, seed=0):
    """Exchangeable-by-construction control for the implementation itself."""
    rng = np.random.default_rng(seed)
    covs = []
    for _ in range(trials):
        p_true = rng.beta(2.0, 1.0, n)
        k = rng.integers(0, 2, n)
        S = np.empty((n, 2)); S[np.arange(n), k] = p_true
        S[np.arange(n), 1 - k] = 1 - p_true
        Y = np.zeros((n, 2), bool); Y[np.arange(n), k] = True
        cal = rng.permutation(n)[: n // 2]; te = np.setdiff1d(np.arange(n), cal)
        t = split_conformal_threshold(S[cal], Y[cal], alpha)
        covs.append(coverage(S[te], Y[te], t))
    covs = np.array(covs)
    return {"n_per_trial": n, "trials": trials, "target": NOMINAL,
            "mean_coverage": float(covs.mean()),
            "sd_coverage": float(covs.std()),
            "frac_trials_below_target": float((covs < NOMINAL).mean())}


def sanity_crc(S, Y, y, g, rng):
    """The module's two thresholding routines should agree on this problem:
    with one-hot Y the per-window FNR is 1 - covered, so conformal risk control
    at alpha and split conformal at alpha bound the same quantity."""
    cal = np.zeros(len(y), bool)
    for grp in np.unique(g):
        idx = np.where(g == grp)[0]
        cal[rng.permutation(idx)[: len(idx) // 2]] = True
        te = ~cal
    te = ~cal
    t_split = split_conformal_threshold(S[cal], Y[cal], ALPHA)
    t_crc = conformal_risk_control(S[cal], Y[cal], ALPHA)
    return {"t_split_conformal": float(t_split), "t_crc": float(t_crc),
            "test_coverage_split": float(coverage(S[te], Y[te], t_split)),
            "test_coverage_crc": float(coverage(S[te], Y[te], t_crc)),
            "test_fnr_split": float(false_negative_rate(S[te], Y[te], t_split)),
            "test_fnr_crc": float(false_negative_rate(S[te], Y[te], t_crc)),
            "test_set_size_split": float(mean_set_size(S[te], t_split))}


# ---------------------------------------------------------------- shift
def shift_case(layer):
    z = np.load(os.path.join(OUT, f"shift_L{layer}.npz"), allow_pickle=True)
    out = {}
    for src, tgt in (("dogs", "cats"), ("cats", "dogs")):
        pre = f"{src}2{tgt}"
        calp, calm = z[f"{pre}_calp"], z[f"{pre}_calmask"]
        tgtp, sy = z[f"{pre}_tgtp"], z[f"{pre}_srcy"]
        ty, tg = z[f"{pre}_tgty"], z[f"{pre}_tgtg"]
        acc = Acc(len(ty)); acc_src = Acc(len(sy))
        ths = []
        for r in range(calp.shape[0]):
            if not np.isfinite(calp[r]).all():
                continue
            Sc, Yc = build(calp[r], sy)
            m = calm[r]
            if len(np.unique(sy[m])) < 2:
                continue
            t = split_conformal_threshold(Sc[m], Yc[m], ALPHA)
            ths.append(t)
            St, Yt = build(tgtp[r], ty)
            acc.add(np.arange(len(ty)), St, Yt, ty, t)
            # in-domain reference: held-out SOURCE clips are not used for cal
            # here, so re-split the source cal groups in half for an honest
            # same-species number
            ci = np.where(m)[0]
            half = len(ci) // 2
            perm = np.random.default_rng(r).permutation(ci)
            c1, c2 = perm[:half], perm[half:]
            if len(np.unique(sy[c1])) < 2:
                continue
            t2 = split_conformal_threshold(Sc[c1], Yc[c1], ALPHA)
            acc_src.add(c2, Sc, Yc, sy, t2)
        gsrc = np.array(["src"] * len(sy))
        out[pre] = {
            "mean_threshold": float(np.mean(ths)),
            "target": summarize(acc, ty, tg, None),
            "source_in_domain": {
                "marginal_coverage": float(acc_src.cov[acc_src.cnt > 0].sum()
                                           / acc_src.cnt[acc_src.cnt > 0].sum()),
                "marginal_set_size": float(acc_src.size[acc_src.cnt > 0].sum()
                                           / acc_src.cnt[acc_src.cnt > 0].sum()),
            },
        }
    return out


# ---------------------------------------------------------------- main
def main():
    res = {
        "config": {"alpha": ALPHA, "nominal_coverage": NOMINAL,
                   "layers": list(LAYERS), "n_splits": R_SPLITS,
                   "encoder": "WavLM-base-plus (cached embeddings)",
                   "probe": "LogisticRegression(C=1, class_weight=balanced), "
                            "leave-one-group-out out-of-fold probabilities",
                   "min_group_cal_for_own_threshold": MIN_GROUP_CAL,
                   "conformal_module": "/Users/vish/esp-lab/conformal_pam/src/conformal.py"},
        "datasets": {}, "shift": {}, "sanity": {},
    }

    res["sanity"]["synthetic_exchangeable"] = sanity_synthetic()
    print("synthetic exchangeable control:",
          res["sanity"]["synthetic_exchangeable"]["mean_coverage"], flush=True)

    for layer in LAYERS:
        z = np.load(os.path.join(OUT, f"oof_L{layer}.npz"), allow_pickle=True)
        for name in DATASETS:
            p1 = z[f"{name}_p1"]; y = z[f"{name}_y"].astype(int)
            g = z[f"{name}_g"]; ctx = z[f"{name}_ctx"]
            S, Y = build(p1, y)
            rng = np.random.default_rng(1234)
            aA, n_used = design_A(S, Y, y, g, rng)
            aB = design_B(S, Y, y, g, rng)
            blk = {
                "n_clips": int(len(y)), "n_groups": int(len(np.unique(g))),
                "oof_accuracy": float(((p1 >= .5).astype(int) == y).mean()),
                "n_group_disjoint_splits": n_used,
                "designA_group_disjoint": {k: summarize(v, y, g, ctx)
                                           for k, v in aA.items()},
                "designB_within_group": {k: summarize(v, y, g, ctx)
                                         for k, v in aB.items()},
            }
            if layer == LAYERS[0]:
                blk["sanity_crc_vs_split"] = sanity_crc(
                    S, Y, y, g, np.random.default_rng(7))
            res["datasets"].setdefault(name, {})[f"L{layer}"] = blk
            a = blk["designA_group_disjoint"]["pooled"]
            b = blk["designB_within_group"]["pooled"]
            c = blk["designB_within_group"]["mondrian_group"]
            print(f"L{layer} {name:5s} A/pooled marg={a['marginal_coverage']:.3f} "
                  f"min={a['coverage_spread']['min']:.3f} "
                  f"iqr={a['coverage_spread']['iqr']:.3f} "
                  f"<0.80: {a['n_groups_below_0.80']}/{a['n_groups']} | "
                  f"B/pooled marg={b['marginal_coverage']:.3f} "
                  f"min={b['coverage_spread']['min']:.3f} | "
                  f"B/group marg={c['marginal_coverage']:.3f} "
                  f"min={c['coverage_spread']['min']:.3f}", flush=True)

        res["shift"][f"L{layer}"] = shift_case(layer)
        for k, v in res["shift"][f"L{layer}"].items():
            print(f"L{layer} shift {k}: target marg={v['target']['marginal_coverage']:.3f} "
                  f"min={v['target']['coverage_spread']['min']:.3f} "
                  f"size={v['target']['marginal_set_size']:.2f} | "
                  f"in-domain {v['source_in_domain']['marginal_coverage']:.3f}",
                  flush=True)

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("\nwrote", os.path.join(OUT, "results.json"))


if __name__ == "__main__":
    main()
