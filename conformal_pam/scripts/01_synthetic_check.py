"""
Does the guarantee actually hold? Check it on synthetic data before touching
real audio.

Runs three experiments:

  A. No shift          -- every method should hit its nominal coverage.
  B. Site shift        -- calibrate on site A, test on site B. This is where
                          temperature scaling is expected to break and
                          conformal is expected to survive.
  C. Long tail         -- rare species. Marginal coverage looks fine while
                          rare species are quietly dropped; Mondrian fixes it.

Nothing here needs a GPU, a download, or torch. It runs in seconds on the
laptop with the numpy/scipy/sklearn that are already installed.

    python conformal_pam/scripts/01_synthetic_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conformal import (  # noqa: E402
    conformal_risk_control,
    coverage,
    expected_calibration_error,
    false_negative_rate,
    mean_set_size,
    mondrian_thresholds,
    split_conformal_threshold,
    temperature_scale,
    weighted_conformal_threshold,
)

ALPHA = 0.10
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------
# a toy detector whose miscalibration we control
# --------------------------------------------------------------------------

def simulate_site(
    n: int,
    K: int = 40,
    prevalence: np.ndarray | None = None,
    separation: float = 2.2,
    logit_shift: float = 0.0,
    logit_scale: float = 1.0,
    rng: np.random.Generator = RNG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (scores, logits, labels) for one recording site.

    ``separation`` is how well the detector discriminates. ``logit_shift`` and
    ``logit_scale`` are the domain shift knobs: a new site with a different
    microphone and noise floor makes the detector systematically
    over-confident (shift > 0) and/or differently spread (scale != 1).
    """
    if prevalence is None:
        prevalence = np.full(K, 0.05)

    labels = rng.random((n, K)) < prevalence[None, :]

    logits = rng.normal(-separation, 1.0, size=(n, K))
    logits[labels] = rng.normal(separation * 0.5, 1.0, size=int(labels.sum()))

    logits = logits * logit_scale + logit_shift
    scores = 1.0 / (1.0 + np.exp(-logits))
    return scores, logits, labels


def line(name: str, cov: float, fnr: float, size: float, target: float) -> str:
    ok = "OK " if cov >= target - 0.02 else "MISS"
    return (f"  {ok} {name:<34s} coverage={cov:6.3f}  "
            f"FNR={fnr:6.3f}  set size={size:6.2f}")


# --------------------------------------------------------------------------
# A. no shift
# --------------------------------------------------------------------------

def experiment_a() -> None:
    print("\nA. No distribution shift  (target coverage >= %.2f)" % (1 - ALPHA))
    print("   " + "-" * 66)

    s, z, y = simulate_site(8000)
    cal, tst = slice(0, 4000), slice(4000, 8000)

    t_split = split_conformal_threshold(s[cal], y[cal], ALPHA)
    t_crc = conformal_risk_control(s[cal], y[cal], ALPHA)
    t_mond = mondrian_thresholds(s[cal], y[cal], ALPHA)

    T = temperature_scale(z[cal], y[cal])
    s_temp = 1.0 / (1.0 + np.exp(-z / T))
    t_temp = 0.5  # the conventional choice, with no guarantee attached

    for nm, sc, th in [
        ("split conformal", s, t_split),
        ("conformal risk control (FNR)", s, t_crc),
        ("Mondrian (per species)", s, t_mond),
        ("temperature scaling @ 0.5", s_temp, t_temp),
    ]:
        print(line(nm, coverage(sc[tst], y[tst], th),
                   false_negative_rate(sc[tst], y[tst], th),
                   mean_set_size(sc[tst], th), 1 - ALPHA))

    print(f"     [temperature fitted T = {T:.3f}; "
          f"ECE before = {expected_calibration_error(s[cal], y[cal]):.4f}, "
          f"after = {expected_calibration_error(s_temp[cal], y[cal]):.4f}]")


# --------------------------------------------------------------------------
# B. site shift -- the headline experiment
# --------------------------------------------------------------------------

def experiment_b() -> None:
    print("\nB. Site shift: calibrate on site A, deploy on site B")
    print("   site B is noisier: the detector is LESS confident and less separable,")
    print("   which is the failure that actually costs an ecologist true detections.")
    print("   " + "-" * 66)

    sA, zA, yA = simulate_site(6000, logit_shift=0.0, logit_scale=1.0)
    sB, zB, yB = simulate_site(6000, separation=1.5, logit_shift=-0.8, logit_scale=0.9)

    t_split = split_conformal_threshold(sA, yA, ALPHA)
    t_crc = conformal_risk_control(sA, yA, ALPHA)

    T = temperature_scale(zA, yA)
    sB_temp = 1.0 / (1.0 + np.exp(-zB / T))

    # Weighted conformal needs a likelihood ratio. In a real study you would
    # fit a domain classifier on embeddings; here we fit one on the score
    # vectors themselves, which is the same procedure with weaker features.
    from sklearn.linear_model import LogisticRegression

    X = np.vstack([sA, sB])
    d = np.concatenate([np.zeros(len(sA)), np.ones(len(sB))])
    clf = LogisticRegression(max_iter=1000).fit(X, d)
    p = np.clip(clf.predict_proba(sA)[:, 1], 1e-4, 1 - 1e-4)
    w = p / (1.0 - p)

    t_w = weighted_conformal_threshold(sA, yA, w, ALPHA)

    for nm, sc, th in [
        ("split conformal", sB, t_split),
        ("conformal risk control (FNR)", sB, t_crc),
        ("weighted conformal", sB, t_w),
        ("temperature scaling @ 0.5", sB_temp, 0.5),
    ]:
        print(line(nm, coverage(sc, yB, th), false_negative_rate(sc, yB, th),
                   mean_set_size(sc, th), 1 - ALPHA))

    print(f"     [mean likelihood ratio {w.mean():.3f}; "
          f"thresholds: split {t_split:.3f}, weighted {t_w:.3f}]")


# --------------------------------------------------------------------------
# C. long tail
# --------------------------------------------------------------------------

def experiment_c() -> None:
    print("\nC. Long-tailed species: does marginal coverage hide rare-species failure?")
    print("   " + "-" * 66)

    K = 40
    prev = np.geomspace(0.30, 0.002, K)          # a few common, many rare

    # Rare species are also HARDER: fewer training examples upstream means the
    # detector separates them less well. Simulating equal difficulty would hide
    # the very effect Mondrian exists to fix.
    sep = np.linspace(2.6, 0.9, K)
    parts = [simulate_site(12000, K=1, prevalence=prev[k:k + 1], separation=sep[k])
             for k in range(K)]
    s = np.hstack([p[0] for p in parts])
    y = np.hstack([p[2] for p in parts])
    cal, tst = slice(0, 6000), slice(6000, 12000)

    t_split = split_conformal_threshold(s[cal], y[cal], ALPHA)
    t_mond = mondrian_thresholds(s[cal], y[cal], ALPHA, min_count=20)

    rare = np.arange(K) >= K // 2                # the rarer half

    def cov_subset(th, cols):
        m = np.zeros_like(y[tst], dtype=bool)
        m[:, cols] = y[tst][:, cols]
        return coverage(s[tst], m, th)

    print(f"  split conformal    overall={coverage(s[tst], y[tst], t_split):.3f}  "
          f"rare half={cov_subset(t_split, rare):.3f}")
    print(f"  Mondrian           overall={coverage(s[tst], y[tst], t_mond):.3f}  "
          f"rare half={cov_subset(t_mond, rare):.3f}")

    n_fallback = int((t_mond == t_split).sum())
    print(f"     [{n_fallback}/{K} species had <20 calibration positives and fell "
          f"back to the pooled threshold]")


def experiment_d() -> None:
    """If shift breaks the guarantee, how little target-site labelling restores it?

    This is the practitioner-facing version of the question. Nobody can label a
    whole new site, but labelling 50 windows is an afternoon. The useful output
    is a number: the smallest calibration set at the new site that brings
    coverage back to nominal.
    """
    print("\nD. How much labelling at the NEW site restores the guarantee?")
    print("   " + "-" * 66)

    sA, _, yA = simulate_site(6000, logit_shift=0.0, logit_scale=1.0)
    sB, _, yB = simulate_site(8000, separation=1.5, logit_shift=-0.8, logit_scale=0.9)

    t_source = split_conformal_threshold(sA, yA, ALPHA)
    print(f"  calibrate on source site only      coverage on B = "
          f"{coverage(sB, yB, t_source):.3f}   set size={mean_set_size(sB, t_source):5.2f}")

    held = slice(4000, 8000)                      # untouched evaluation half of site B
    for n in (25, 50, 100, 250, 500, 1000, 2000):
        t = split_conformal_threshold(sB[:n], yB[:n], ALPHA)
        cov = coverage(sB[held], yB[held], t)
        size = mean_set_size(sB[held], t)
        n_pos = int(yB[:n].sum())
        flag = "OK " if cov >= 1 - ALPHA - 0.02 else "   "
        print(f"  {flag}+{n:5d} labelled windows from B      coverage on B = "
              f"{cov:.3f}   set size={size:5.2f}   ({n_pos} positive pairs)")


if __name__ == "__main__":
    print("=" * 70)
    print("Conformal uncertainty for PAM -- synthetic validation")
    print(f"alpha = {ALPHA}   (so we want >= {1 - ALPHA:.0%} of true detections kept)")
    print("=" * 70)
    experiment_a()
    experiment_b()
    experiment_c()
    experiment_d()
    print("\n" + "=" * 70)
    print("Read it like this: conformal methods should sit at or just above the")
    print("target in A and stay there in B. A fixed 0.5 threshold has no")
    print("guarantee and should visibly drift once the site changes.")
    print("=" * 70)
