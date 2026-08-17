"""
Distribution-free uncertainty for passive acoustic monitoring.

Pure numpy/scipy. No torch, no GPU, no downloads.

The setting
-----------
A bioacoustic detector scores each audio window against K species. You get a
score matrix S (n_windows, K) and, for evaluation, a binary label matrix
Y (n_windows, K). Multiple species can be present in one window, so this is
MULTI-LABEL, not multi-class.

The practitioner's real question is not "what is the accuracy" but:
    "If I only have budget to listen to some of these detections, what can I
     guarantee about what I'm missing?"

Conformal prediction answers exactly that, with a finite-sample guarantee that
holds for ANY model, with no distributional assumptions beyond exchangeability
between the calibration set and the test set.

Methods implemented
-------------------
split_conformal_threshold : marginal coverage over labels        (Vovk et al.)
conformal_risk_control    : bound the expected FNR at level alpha (Angelopoulos
                            et al., "Conformal Risk Control", ICLR 2024,
                            arXiv:2208.02814)
mondrian_thresholds       : per-class thresholds for long-tailed species
weighted_conformal_threshold : covariate shift via likelihood-ratio weights
                            (Tibshirani, Barber, Candès, Ramdas, NeurIPS 2019)

Baselines: temperature_scale, platt_scale.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

__all__ = [
    "conformal_quantile",
    "split_conformal_threshold",
    "conformal_risk_control",
    "mondrian_thresholds",
    "weighted_conformal_threshold",
    "temperature_scale",
    "platt_scale",
    "predict_sets",
    "coverage",
    "mean_set_size",
    "false_negative_rate",
    "expected_calibration_error",
]


# --------------------------------------------------------------------------
# the quantile that makes the guarantee work
# --------------------------------------------------------------------------

def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The conformal quantile of calibration scores.

    Returns the ``ceil((n+1)(1-alpha)) / n`` empirical quantile. The +1 is the
    whole trick: it accounts for the test point being exchangeable with the
    calibration points, and is what upgrades an asymptotic statement into a
    finite-sample one.

    Guarantee (Vovk et al.): for exchangeable data,

        P(s(X_test, Y_test) <= qhat) >= 1 - alpha

    Returns ``+inf`` when n is too small for the requested alpha -- meaning
    "you cannot make this guarantee with this little calibration data", which
    is the honest answer rather than a silently invalid threshold.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = scores.size
    if n == 0:
        return np.inf

    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    if level > 1.0:
        # n < 1/alpha - 1: not enough calibration points for this alpha.
        return np.inf
    return float(np.quantile(scores, level, method="higher"))


# --------------------------------------------------------------------------
# 1. split conformal
# --------------------------------------------------------------------------

def split_conformal_threshold(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Single global score threshold with marginal coverage >= 1 - alpha.

    Nonconformity score for a *present* species is ``1 - s`` : a true detection
    the model scored low is nonconforming. We take the conformal quantile over
    all positive (window, species) pairs, then convert back to a score
    threshold.

    Parameters
    ----------
    cal_scores, cal_labels : (n, K) arrays. Scores in [0, 1], labels in {0, 1}.

    Returns
    -------
    Score threshold ``t``. Predict species k present iff ``s[:, k] >= t``.
    Species actually present are then retained with probability >= 1 - alpha.
    """
    cal_scores = np.asarray(cal_scores, dtype=float)
    cal_labels = np.asarray(cal_labels).astype(bool)

    pos = cal_scores[cal_labels]              # scores the model gave to true positives
    if pos.size == 0:
        return 0.0

    qhat = conformal_quantile(1.0 - pos, alpha)
    if not np.isfinite(qhat):
        return 0.0                            # cannot guarantee -> keep everything
    return float(np.clip(1.0 - qhat, 0.0, 1.0))


# --------------------------------------------------------------------------
# 2. conformal risk control  -- the one that matches what ecologists ask for
# --------------------------------------------------------------------------

def _fnr_per_window(scores: np.ndarray, labels: np.ndarray, t: float) -> np.ndarray:
    """Per-window fraction of truly-present species missed at threshold t.

    Windows with no species present contribute loss 0 (nothing to miss).
    """
    kept = scores >= t
    n_true = labels.sum(axis=1)
    hit = (kept & labels).sum(axis=1)

    loss = np.zeros(scores.shape[0], dtype=float)
    nz = n_true > 0
    loss[nz] = 1.0 - hit[nz] / n_true[nz]
    return loss


def conformal_risk_control(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    alpha: float = 0.1,
    n_grid: int = 400,
) -> float:
    """Threshold whose expected false-negative rate is provably <= alpha.

    This is the practitioner-facing method. The guarantee is on the *risk*, not
    on coverage of a set:

        E[ L(threshold) ] <= alpha        (Angelopoulos et al., ICLR 2024)

    where ``L`` is the per-window FNR. It holds in expectation over the draw of
    the calibration set, for any bounded loss that is monotone in the
    parameter.

    Monotonicity: as ``t`` decreases the retained set grows, so FNR is
    non-decreasing in ``t``. The loss is bounded by B = 1.

    .. warning::
       This construction is valid for FNR and **invalid for FDR**. FDR is not
       monotone in the threshold in either direction -- adding a species can
       *lower* FDR when that species is a true positive. Concretely, with
       ``Y = {0}`` and ``s = (0.90, 0.95, 0.70)``: FDR is 1.000 at t=0.95,
       drops to 0.500 at t=0.90, and rises to 0.667 at t=0.70. Proposition 2
       of the CRC paper shows the guarantee fails for non-monotone losses. If
       you want FDR control, monotonise the loss first (Corollary 1) or use
       Learn-then-Test. Do not simply swap the loss function here.

    We therefore scan
    ``t`` downward and take the LARGEST ``t`` (i.e. the smallest, cheapest
    detection set) that still satisfies the CRC condition

        ( n * Rhat(t) + B ) / (n + 1)  <=  alpha

    Returns
    -------
    Score threshold ``t``. Reviewing every detection with ``s >= t`` misses at
    most an ``alpha`` fraction of true detections, in expectation.
    """
    cal_scores = np.asarray(cal_scores, dtype=float)
    cal_labels = np.asarray(cal_labels).astype(bool)
    n = cal_scores.shape[0]
    B = 1.0

    if n == 0 or cal_labels.sum() == 0:
        return 0.0

    # The condition is only ever satisfiable if alpha > B/(n+1).
    if alpha <= B / (n + 1):
        return 0.0

    # Candidate thresholds: the observed positive scores are the only points
    # where the empirical risk changes.
    pos = cal_scores[cal_labels]
    grid = np.unique(np.quantile(pos, np.linspace(0.0, 1.0, n_grid)))
    grid = np.concatenate([grid, [0.0]])
    grid = np.sort(np.unique(grid))[::-1]          # descending: large t first

    for t in grid:
        rhat = _fnr_per_window(cal_scores, cal_labels, t).mean()
        if (n * rhat + B) / (n + 1) <= alpha:
            return float(t)
    return 0.0


# --------------------------------------------------------------------------
# 3. Mondrian / class-conditional
# --------------------------------------------------------------------------

def mondrian_thresholds(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    alpha: float = 0.1,
    min_count: int = 20,
) -> np.ndarray:
    """One threshold per species, so coverage holds *within* each species.

    Marginal coverage is a weak promise on long-tailed data: you can hit 90%
    overall while systematically dropping every rare species, which is usually
    the exact opposite of what an ecologist wants. Class-conditional (Mondrian)
    conformal fixes each species separately.

    The cost is sample size. A species with 5 calibration examples cannot
    support a 90% guarantee -- ``ceil((n+1)(1-alpha))/n > 1``. Species with
    fewer than ``min_count`` positives fall back to the pooled threshold, and
    that fallback must be reported honestly in any paper: those species get the
    marginal guarantee, not the conditional one.

    Returns
    -------
    (K,) array of per-species thresholds.
    """
    cal_scores = np.asarray(cal_scores, dtype=float)
    cal_labels = np.asarray(cal_labels).astype(bool)
    K = cal_scores.shape[1]

    pooled = split_conformal_threshold(cal_scores, cal_labels, alpha)
    out = np.full(K, pooled, dtype=float)

    for k in range(K):
        pos = cal_scores[cal_labels[:, k], k]
        if pos.size < min_count:
            continue
        qhat = conformal_quantile(1.0 - pos, alpha)
        if np.isfinite(qhat):
            out[k] = float(np.clip(1.0 - qhat, 0.0, 1.0))
    return out


# --------------------------------------------------------------------------
# 4. weighted conformal, for covariate shift between sites
# --------------------------------------------------------------------------

def weighted_conformal_threshold(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Split conformal reweighted by a likelihood ratio, for a shifted test site.

    Exchangeability breaks when calibration and test audio come from different
    sites: different microphones, background noise, species mix. Weighted
    conformal restores the guarantee *if* you know the covariate-shift
    likelihood ratio ``w(x) = dP_test/dP_cal``.

    In practice you estimate ``w`` by training a domain classifier to separate
    calibration from test windows on their embeddings, then using
    ``w = p/(1-p)``. The guarantee is only as good as that estimate -- this is
    the method's real failure mode, and it should be stated as such rather than
    presented as free robustness.

    Parameters
    ----------
    weights : (n,) non-negative per-calibration-window likelihood ratios.
    """
    cal_scores = np.asarray(cal_scores, dtype=float)
    cal_labels = np.asarray(cal_labels).astype(bool)
    weights = np.asarray(weights, dtype=float)

    if cal_labels.sum() == 0:
        return 0.0

    # lift window weights onto (window, species) positive pairs
    w_pairs = np.repeat(weights[:, None], cal_scores.shape[1], axis=1)[cal_labels]
    s_pos = 1.0 - cal_scores[cal_labels]

    order = np.argsort(s_pos)
    s_sorted = s_pos[order]
    w_sorted = w_pairs[order]

    total = w_sorted.sum()
    if total <= 0:
        return 0.0

    # The test point carries mass 1 in the normalisation -- the weighted
    # analogue of the (n+1) in the unweighted quantile.
    cum = np.cumsum(w_sorted) / (total + 1.0)
    idx = np.searchsorted(cum, 1.0 - alpha, side="left")
    if idx >= s_sorted.size:
        return 0.0
    return float(np.clip(1.0 - s_sorted[idx], 0.0, 1.0))


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------

def temperature_scale(cal_logits: np.ndarray, cal_labels: np.ndarray) -> float:
    """Fit a single temperature by minimising multi-label NLL. Returns T."""
    cal_logits = np.asarray(cal_logits, dtype=float)
    y = np.asarray(cal_labels, dtype=float)

    def nll(log_t: float) -> float:
        z = cal_logits / np.exp(log_t)
        # numerically stable elementwise binary cross-entropy
        return float(np.mean(np.logaddexp(0.0, z) - y * z))

    res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(res.x))


def platt_scale(cal_logits: np.ndarray, cal_labels: np.ndarray) -> tuple[float, float]:
    """Fit scalar a, b in sigmoid(a * z + b) by minimising NLL."""
    from scipy.optimize import minimize

    cal_logits = np.asarray(cal_logits, dtype=float)
    y = np.asarray(cal_labels, dtype=float)

    def nll(p: np.ndarray) -> float:
        z = p[0] * cal_logits + p[1]
        return float(np.mean(np.logaddexp(0.0, z) - y * z))

    res = minimize(nll, x0=np.array([1.0, 0.0]), method="Nelder-Mead")
    return float(res.x[0]), float(res.x[1])


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def predict_sets(scores: np.ndarray, threshold) -> np.ndarray:
    """Boolean (n, K) prediction sets. ``threshold`` is scalar or (K,)."""
    return np.asarray(scores, dtype=float) >= np.asarray(threshold, dtype=float)


def coverage(scores: np.ndarray, labels: np.ndarray, threshold) -> float:
    """Fraction of truly-present species that are retained."""
    labels = np.asarray(labels).astype(bool)
    if labels.sum() == 0:
        return float("nan")
    return float(predict_sets(scores, threshold)[labels].mean())


def mean_set_size(scores: np.ndarray, threshold) -> float:
    """Average number of species retained per window -- the human review cost."""
    return float(predict_sets(scores, threshold).sum(axis=1).mean())


def false_negative_rate(scores: np.ndarray, labels: np.ndarray, threshold) -> float:
    """Mean per-window FNR -- the quantity conformal_risk_control bounds."""
    labels = np.asarray(labels).astype(bool)
    kept = predict_sets(scores, threshold)
    n_true = labels.sum(axis=1)
    hit = (kept & labels).sum(axis=1)
    nz = n_true > 0
    if not nz.any():
        return float("nan")
    return float((1.0 - hit[nz] / n_true[nz]).mean())


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Binned ECE over all (window, species) pairs.

    Reported because reviewers expect it, with the caveat that ECE is
    bin-sensitive and can be driven to near-zero by the overwhelming majority
    of true negatives in sparse multi-label data. It is a weaker diagnostic
    than coverage under shift, which is precisely the argument for conformal.
    """
    p = np.asarray(probs, dtype=float).ravel()
    y = np.asarray(labels, dtype=float).ravel()

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.mean()) * abs(y[m].mean() - p[m].mean())
    return float(ece)
