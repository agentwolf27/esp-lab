"""
STAGE 2 (laptop, run forever): score matrix -> the result.

Takes the .npz produced by 02_extract_scores_cloud.py and runs the real version
of the synthetic experiment. Needs only numpy/scipy/sklearn/matplotlib, all of
which are already installed.

    python conformal_pam/scripts/03_real_analysis.py birdset_perch_scores.npz

The experiment
--------------
Calibrate on site A, deploy on site B, and ask three questions:

  1. Does the guarantee survive the site change?  (synthetic says: no)
  2. How much labelling at site B restores it?    (synthetic says: ~50-100 windows)
  3. Do rare species fail while the average looks fine? (synthetic says: yes,
     and Mondrian fixes it)

The one rule that makes it honest: split by RECORDING, never randomly. Windows
from the same audio file are not independent -- the same individual bird, the
same background, often seconds apart. A random split lets near-duplicates sit
in both calibration and test, which inflates every number and is the single
most common way this kind of study is quietly wrong.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conformal import (  # noqa: E402
    conformal_risk_control,
    coverage,
    false_negative_rate,
    mean_set_size,
    mondrian_thresholds,
    split_conformal_threshold,
    temperature_scale,
)

ALPHA = 0.10


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def load(path: str):
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    print(f"loaded {path}")
    print(f"  {d['logits'].shape[0]:,} windows x {d['logits'].shape[1]} species")
    print(f"  model={meta['model']}  sites={meta['sites']}  "
          f"affine_applied={meta.get('affine_applied')}")
    return d["logits"], d["labels"], d["recording"], d["site"], meta


def by_recording(rec: np.ndarray, frac: float, seed: int = 0):
    """Split indices by recording id, not by row."""
    uniq = np.unique(rec)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    cut = int(len(uniq) * frac)
    first = set(uniq[:cut])
    m = np.array([r in first for r in rec])
    return m, ~m


def row(name: str, cov: float, fnr: float, size: float) -> None:
    flag = "OK " if cov >= 1 - ALPHA - 0.02 else "MISS"
    print(f"  {flag} {name:<32s} coverage={cov:6.3f}  FNR={fnr:6.3f}  "
          f"set size={size:6.2f}")


def main(path: str) -> None:
    logits, Y, rec, site, meta = load(path)
    S = sigmoid(logits)
    sites = sorted(set(site.tolist()))

    print(f"\npositives: {int(Y.sum()):,}   recordings: {len(set(rec.tolist())):,}")

    # ---------------------------------------------------------------
    print("\n" + "=" * 68)
    print("1. WITHIN SITE  (calibrate and deploy on the same site)")
    print("=" * 68)
    for s in sites:
        m = site == s
        cal, tst = by_recording(rec[m], 0.5)
        Ss, Ys = S[m], Y[m]
        if Ys[cal].sum() < 50:
            print(f"  {s}: too few positives to calibrate")
            continue
        print(f"\n  site {s}  ({m.sum():,} windows, {int(Ys.sum()):,} positives)")
        for nm, th in [
            ("split conformal", split_conformal_threshold(Ss[cal], Ys[cal], ALPHA)),
            ("conformal risk control", conformal_risk_control(Ss[cal], Ys[cal], ALPHA)),
            ("Mondrian", mondrian_thresholds(Ss[cal], Ys[cal], ALPHA)),
            ("fixed 0.5", 0.5),
        ]:
            row(nm, coverage(Ss[tst], Ys[tst], th),
                false_negative_rate(Ss[tst], Ys[tst], th), mean_set_size(Ss[tst], th))

    if len(sites) < 2:
        print("\nOnly one site present -- add a second to run the shift experiment.")
        return

    # ---------------------------------------------------------------
    A, B = sites[0], sites[1]
    mA, mB = site == A, site == B
    print("\n" + "=" * 68)
    print(f"2. ACROSS SITES  (calibrate on {A}, deploy on {B})")
    print("=" * 68)

    t_split = split_conformal_threshold(S[mA], Y[mA], ALPHA)
    t_crc = conformal_risk_control(S[mA], Y[mA], ALPHA)
    T = temperature_scale(logits[mA], Y[mA])
    S_temp = sigmoid(logits / T)

    for nm, sc, th in [
        ("split conformal", S, t_split),
        ("conformal risk control", S, t_crc),
        (f"temperature (T={T:.2f}) @0.5", S_temp, 0.5),
        ("fixed 0.5", S, 0.5),
    ]:
        row(nm, coverage(sc[mB], Y[mB], th), false_negative_rate(sc[mB], Y[mB], th),
            mean_set_size(sc[mB], th))

    # ---------------------------------------------------------------
    print("\n" + "=" * 68)
    print(f"3. HOW MUCH LABELLING AT {B} RESTORES THE GUARANTEE?")
    print("=" * 68)
    print(f"  {'(source-only baseline)':<34s} coverage={coverage(S[mB], Y[mB], t_split):6.3f}")

    idxB = np.where(mB)[0]
    calB, tstB = by_recording(rec[idxB], 0.5, seed=1)
    cal_idx, tst_idx = idxB[calB], idxB[tstB]

    for n in (25, 50, 100, 250, 500, 1000):
        if n > len(cal_idx):
            break
        sub = cal_idx[:n]
        t = split_conformal_threshold(S[sub], Y[sub], ALPHA)
        cov = coverage(S[tst_idx], Y[tst_idx], t)
        flag = "OK " if cov >= 1 - ALPHA - 0.02 else "   "
        print(f"  {flag} +{n:5d} windows from {B}          coverage={cov:6.3f}   "
              f"set size={mean_set_size(S[tst_idx], t):6.2f}   "
              f"({int(Y[sub].sum())} positives)")

    # ---------------------------------------------------------------
    print("\n" + "=" * 68)
    print("4. RARE SPECIES  (does the average hide systematic failure?)")
    print("=" * 68)
    cal, tst = by_recording(rec, 0.5, seed=2)
    freq = Y[cal].sum(axis=0)
    rare = freq <= np.median(freq[freq > 0]) if (freq > 0).any() else freq < 0

    t_split = split_conformal_threshold(S[cal], Y[cal], ALPHA)
    t_mond = mondrian_thresholds(S[cal], Y[cal], ALPHA)

    def cov_cols(th, cols):
        m = np.zeros_like(Y[tst])
        m[:, cols] = Y[tst][:, cols]
        return coverage(S[tst], m, th)

    print(f"  split conformal   overall={coverage(S[tst], Y[tst], t_split):.3f}   "
          f"rare={cov_cols(t_split, rare):.3f}")
    print(f"  Mondrian          overall={coverage(S[tst], Y[tst], t_mond):.3f}   "
          f"rare={cov_cols(t_mond, rare):.3f}")
    print(f"     [{int((t_mond == t_split).sum())}/{Y.shape[1]} species fell back "
          f"to the pooled threshold]")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "birdset_perch_scores.npz")
