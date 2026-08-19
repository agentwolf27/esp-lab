"""Paper-sized version of the coverage result: one row, three corpora.

figures/coverage.png is a 5x4 exploration grid -- correct, but unreadable at
workshop column width. This keeps the single panel the paper's claim rests on:
design A (calibrate on OTHER individuals/labs, test on a held-out one), pooled
split conformal, the deployment-realistic case. Marginal coverage lands on the
0.90 target in all three corpora; the per-group spread does not.

    python paper_figure.py            -> figures/coverage_paper.png
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(REPO, "results", "out_conformal", "results.json")
OUT = os.path.join(REPO, "figures", "coverage_paper.png")

LAYER, TARGET, FLOOR = "L9", 0.90, 0.80
GOOD, MID, BAD = "#2c7fb8", "#e6a03c", "#d13b2e"
PANELS = [("cats", "cats · 20 individuals"), ("dogs", "dogs · 10 individuals"),
          ("pigs", "pigs · 6 recording labs")]


def colour(c):
    return BAD if c < FLOOR else (MID if c < TARGET else GOOD)


def main():
    d = json.load(open(RES))
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.05), sharey=True)

    for ax, (key, title) in zip(axes, PANELS):
        p = d["datasets"][key][LAYER]["designA_group_disjoint"]["pooled"]
        per = p["per_group"]
        names = list(per)
        cov = np.array([per[g]["coverage"] for g in names])
        n = np.array([per[g]["n"] for g in names], float)
        rng = np.random.default_rng(0)
        x = rng.uniform(-0.28, 0.28, len(cov))

        ax.axhspan(0.0, FLOOR, color=BAD, alpha=0.045, zorder=0)
        ax.axhline(TARGET, ls="--", c="0.25", lw=1.3, zorder=1)
        ax.axhline(p["marginal_coverage"], c="0.45", lw=1.1, zorder=1)
        ax.scatter(x, cov, s=18 + 90 * n / n.max(),
                   c=[colour(c) for c in cov], alpha=0.9, zorder=3,
                   edgecolors="white", linewidths=0.6)

        w = int(np.argmin(cov))
        ax.annotate(f"{names[w].split('_')[-1]}  {cov[w]:.3f}",
                    (x[w], cov[w]), textcoords="offset points", xytext=(11, -3),
                    fontsize=8.5, color=BAD, va="center")
        ax.set_title(title, fontsize=10.5, pad=7)
        ax.set_xticks([]); ax.set_xlim(-0.75, 0.75); ax.set_ylim(0.30, 1.045)
        ax.text(0.5, 0.045,
                f"marginal {p['marginal_coverage']:.3f}   "
                f"worst {p['worst_group_coverage']:.3f}   "
                f"{p['n_groups_below_0.80']}/{p['n_groups']} below 0.80",
                transform=ax.transAxes, ha="center", fontsize=8.2, color="0.3")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel("coverage of the held-out group", fontsize=10)
    axes[0].text(-0.72, TARGET + 0.012, "0.90 target", fontsize=8.2, color="0.25")
    fig.suptitle("Split conformal at $\\alpha$=0.10, calibrated on other individuals: "
                 "the average is met, the individual is not",
                 fontsize=11.5, y=1.015)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
