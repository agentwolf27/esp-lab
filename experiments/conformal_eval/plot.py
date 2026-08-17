"""Stage 3: coverage.png -- per-group coverage strip plots, three calibration
schemes side by side, with the 0.90 target line."""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

OUT = "/private/tmp/claude-501/-Volumes-SSD/eb358682-62a6-4aab-8b76-0d59bcd47add/scratchpad/ctx/out_conformal"
LAYER = "L9"
TARGET, FLOOR = 0.90, 0.80

GOOD, MID, BAD = "#2c7fb8", "#e6a03c", "#d13b2e"
COLS = [
    ("designA_group_disjoint", "pooled",
     "(a) pooled split conformal"),
    ("designA_group_disjoint", "mondrian_class",
     "(b) Mondrian by class"),
    ("designA_group_disjoint", "mondrian_group",
     "(c) Mondrian by group\nunseen group ⇒ falls back to pooled"),
    ("designB_within_group", "pooled",
     "pooled split conformal\nexchangeable control"),
    ("designB_within_group", "mondrian_group",
     "(c) Mondrian by group\nthe group has its own calibration data"),
]
ROWS = [("cats", "cats\n20 individuals"), ("dogs", "dogs\n10 individuals"),
        ("pigs", "pigs\n6 labs")]


def col(c):
    return BAD if c < FLOOR else (MID if c < TARGET else GOOD)


def strip(ax, per, rng, title=None, note=None):
    """One strip of per-group coverage dots."""
    keys = sorted(per, key=lambda k: per[k]["coverage"])
    cov = np.array([per[k]["coverage"] for k in keys])
    n = np.array([per[k]["n"] for k in keys], float)
    x = rng.uniform(-0.26, 0.26, len(keys))
    ax.axhspan(0, FLOOR, color=BAD, alpha=0.035, lw=0)
    ax.axhline(TARGET, color="0.25", ls="--", lw=1.2, zorder=1)
    ax.axhline(FLOOR, color="0.65", ls=":", lw=1.0, zorder=1)
    ax.scatter(x, cov, s=18 + 42 * np.sqrt(n / n.max()),
               c=[col(c) for c in cov], alpha=0.85, lw=0.6,
               edgecolors="white", zorder=3)
    # worst group callout
    ax.annotate(keys[0].replace("cat_", "").replace("dog_", ""),
                (x[0], cov[0]), textcoords="offset points", xytext=(9, -3),
                fontsize=6.5, color=col(cov[0]), zorder=4)
    ax.set_xlim(-0.55, 0.55)
    ax.set_xticks([])
    if title:
        ax.set_title(title, fontsize=8.0, pad=6, linespacing=1.35)
    if note:
        ax.text(0.5, 0.02, note, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=6.6, color="0.25")


def panel(ax, blk, design, scheme, rng, title=None):
    v = blk[design][scheme]
    per = v["per_group"]
    cs = v["coverage_spread"]
    note = (f"marginal {v['marginal_coverage']:.3f}   "
            f"per-group {cs['min']:.3f}–{cs['max']:.3f}\n"
            f"IQR {cs['iqr']:.3f}   {v['n_groups_below_0.80']}/{v['n_groups']} "
            f"below 0.80   |set| {v['marginal_set_size']:.2f}")
    if scheme == "mondrian_group":
        note += (f"\nown \u03b8 for {v['mean_groups_with_own_threshold']:.0f}"
                 f"/{v['n_groups']} groups \u00b7 rest use pooled")
    strip(ax, per, rng, title, note)
    ax.axhline(v["marginal_coverage"], color="0.15", lw=1.6, alpha=0.55, zorder=2)


def main():
    r = json.load(open(os.path.join(OUT, "results.json")))
    rng = np.random.default_rng(3)

    fig = plt.figure(figsize=(14.6, 11.8))
    gs = fig.add_gridspec(4, 5, height_ratios=[1, 1, 1, 1.05],
                          hspace=0.30, wspace=0.13,
                          left=0.075, right=0.985, top=0.845, bottom=0.05)

    axes = {}
    for i, (ds, rlab) in enumerate(ROWS):
        blk = r["datasets"][ds][LAYER]
        for j, (design, scheme, title) in enumerate(COLS):
            ax = fig.add_subplot(gs[i, j])
            axes[(i, j)] = ax
            panel(ax, blk, design, scheme, rng, title if i == 0 else None)
            ax.set_ylim(0.17, 1.04)
            if j == 0:
                ax.set_ylabel(rlab + f"\nn={blk['n_clips']}, OOF acc "
                              f"{blk['oof_accuracy']:.2f}", fontsize=8.5)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7.5)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)

    # ---- headers over the two calibration regimes
    def header(j0, j1, text, color):
        a, b = axes[(0, j0)], axes[(0, j1)]
        x0 = a.get_position().x0
        x1 = b.get_position().x1
        y = a.get_position().y1
        fig.lines.append(Line2D([x0, x1], [y + 0.048, y + 0.048],
                                transform=fig.transFigure, color=color, lw=2.2))
        fig.text((x0 + x1) / 2, y + 0.054, text, ha="center", va="bottom",
                 fontsize=9.4, color=color, weight="bold")

    header(0, 2, "CALIBRATED ON OTHER individuals / labs   —   deployment: the animal is new",
           "#333333")
    header(3, 4, "CALIBRATED ON THE SAME individual / lab", "#1a6b57")

    # ---- shift row
    sh = r["shift"][LAYER]
    for j, (key, lab) in enumerate((("dogs2cats", "calibrate on DOGS  →  test on CATS"),
                                    ("cats2dogs", "calibrate on CATS  →  test on DOGS"))):
        ax = fig.add_subplot(gs[3, 2 * j: 2 * j + 2])
        v = sh[key]["target"]
        cs = v["coverage_spread"]
        note = (f"marginal {v['marginal_coverage']:.3f}   "
                f"per-group {cs['min']:.2f}–{cs['max']:.2f}   IQR {cs['iqr']:.3f}   "
                f"{v['n_groups_below_0.80']}/{v['n_groups']} below 0.80   "
                f"|set| {v['marginal_set_size']:.2f}\n"
                f"same-species reference: "
                f"{sh[key]['source_in_domain']['marginal_coverage']:.3f}")
        strip(ax, v["per_group"], rng, lab, note)
        ax.axhline(v["marginal_coverage"], color="0.15", lw=1.6, alpha=0.55, zorder=2)
        ax.set_ylim(0.10, 1.035)
        ax.tick_params(labelsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if j == 0:
            ax.set_ylabel("SPECIES SHIFT\nper-group coverage", fontsize=8.5)

    axs = fig.add_subplot(gs[3, 4]); axs.axis("off")
    sy = r["sanity"]["synthetic_exchangeable"]
    cb = r["datasets"]["cats"][LAYER]["designB_within_group"]["pooled"]["marginal_coverage"]
    db = r["datasets"]["dogs"][LAYER]["designB_within_group"]["pooled"]["marginal_coverage"]
    pb = r["datasets"]["pigs"][LAYER]["designB_within_group"]["pooled"]["marginal_coverage"]
    axs.text(0.0, 0.97,
             "sanity: the implementation is fine\n\n"
             f"synthetic exchangeable control\n"
             f"   {sy['trials']} trials, n={sy['n_per_trial']}\n"
             f"   mean coverage {sy['mean_coverage']:.4f}   (target 0.900)\n\n"
             "real scores, exchangeable split\n"
             "(random split inside each group)\n"
             f"   cats {cb:.3f}   dogs {db:.3f}   pigs {pb:.3f}\n\n"
             "Marginal coverage is delivered\nexactly as promised. What is\n"
             "uneven is coverage CONDITIONAL\non the animal — which split\n"
             "conformal never promised.",
             va="top", ha="left", fontsize=7.6, family="monospace",
             transform=axs.transAxes, color="0.2")

    handles = [Line2D([], [], marker="o", ls="", color=GOOD, label="≥ 0.90 (nominal)"),
               Line2D([], [], marker="o", ls="", color=MID, label="0.80 – 0.90"),
               Line2D([], [], marker="o", ls="", color=BAD, label="< 0.80"),
               Line2D([], [], color="0.25", ls="--", label="0.90 target"),
               Line2D([], [], color="0.15", lw=1.6, alpha=0.55, label="marginal coverage")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               fontsize=8.6, bbox_to_anchor=(0.53, 0.936))
    fig.text(0.53, 0.982,
             "Split conformal at α = 0.10 on frozen WavLM layer 9: "
             "marginal coverage is met, per-individual coverage is not",
             ha="center", va="top", fontsize=13.5)
    fig.text(0.53, 0.955,
             "one dot = one individual (cats, dogs) or one recording lab (pigs)   ·   "
             "dot size ∝ number of clips   ·   "
             "pooled over 40 calibration/test splits (pigs: all 20)",
             ha="center", va="top", fontsize=9.0, color="0.35")

    p = os.path.join(OUT, "coverage.png")
    fig.savefig(p, dpi=170, facecolor="white")
    print("wrote", p)


if __name__ == "__main__":
    main()
