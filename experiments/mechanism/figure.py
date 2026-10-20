"""out_sae/shared_direction.png -- cosine matrices + top component selectivities,
plus the one number that explains the whole result (how much of each species'
affect axis is the duration axis)."""
from __future__ import annotations

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT

SP = ["cat", "dog", "pig"]
L = "9"


def mat(block, key):
    M = np.eye(3)
    P = np.full((3, 3), np.nan)
    for i, a in enumerate(SP):
        for j, b in enumerate(SP):
            if i == j:
                continue
            k = f"{a}-{b}" if f"{a}-{b}" in block[key] else f"{b}-{a}"
            M[i, j] = block[key][k]["cos"]
            P[i, j] = block[key][k]["p_greater"]
    return M, P


def draw_mat(ax, M, P, title):
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(3), SP); ax.set_yticks(range(3), SP)
    for i in range(3):
        for j in range(3):
            if i == j:
                ax.text(j, i, "1", ha="center", va="center", color="w", fontsize=11)
            else:
                ax.text(j, i, f"{M[i,j]:+.3f}\np={P[i,j]:.3f}", ha="center", va="center",
                        fontsize=9, color="k")
    ax.set_title(title, fontsize=10)
    return im


def main():
    t12 = json.load(open(os.path.join(OUT, "task12.json")))
    t3 = json.load(open(os.path.join(OUT, "task3.json")))
    fu = json.load(open(os.path.join(OUT, "followup.json")))
    z = np.load(os.path.join(OUT, "codes.npz"))
    AUC = z["auc"]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), layout="constrained")
    fig.suptitle("Is there a shared affect direction across cat / dog / pig?  "
                 "WavLM layer 9, within-species z-scored", fontsize=12.5)

    b9 = t12["task1"][L]
    M, P = mat(b9, "raw")
    im = draw_mat(axes[0, 0], M, P,
                  "cos(w_a, w_b) of affect directions\n"
                  "p = within-animal/-team label permutation (200)")
    Mr, Pr = mat(b9, "dur_residualised")
    draw_mat(axes[0, 1], Mr, Pr,
             "same, after regressing log-duration\nout of every embedding dimension")
    fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.8, pad=0.02, label="cosine similarity")

    # null band annotation
    for ax, blk in [(axes[0, 0], b9["raw"]), (axes[0, 1], b9["dur_residualised"])]:
        sds = [blk[k]["null_sd"] for k in blk]
        ax.set_xlabel(f"null SD {min(sds):.2f}-{max(sds):.2f}  "
                      f"(the cosines are inside the null)", fontsize=8)

    # ---- how much of each affect axis is the duration axis
    ax = axes[1, 0]
    layers = ["3", "9", "12"]
    w = 0.26
    x = np.arange(3)
    for k, li in enumerate(layers):
        v = [t12["task1"][li]["cos_w_dur"][s] for s in SP]
        ax.bar(x + (k - 1) * w, v, w, label=f"layer {li}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, SP); ax.set_ylim(0, 1)
    ax.set_ylabel("cos(affect direction, duration direction)")
    ax.set_title("How much of the affect axis is just call duration", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    rel = t12["task1"][L]["split_half_reliability"]
    ax.text(0.02, 0.96, "split-half reliability of w_s (layer 9):\n"
            + "   ".join(f"{s} {rel[s]:.2f}" for s in SP)
            + "\n-> cat/dog axes are only moderately estimable;\n"
              "    the pig axis is highly reliable and is duration",
            transform=ax.transAxes, va="top", fontsize=7.6,
            bbox=dict(fc="#f4f4f4", ec="#bbb", boxstyle="round,pad=0.4"))

    # ---- top component selectivities
    ax = axes[1, 1]
    dev = np.abs(AUC - 0.5).max(0)
    top = np.argsort(dev)[::-1][:10]
    order = top[np.argsort(-np.abs(AUC - 0.5).max(0)[top])]
    xs = np.arange(len(order))
    cols = {"cat": "#4C72B0", "dog": "#DD8452", "pig": "#55A868"}
    for k, s in enumerate(SP):
        ax.bar(xs + (k - 1) * 0.27, AUC[k, order], 0.27, label=s, color=cols[s])
    ax.axhline(0.5, color="k", lw=0.8)
    for t, ls in [(0.6, "--"), (0.4, "--")]:
        ax.axhline(t, color="crimson", lw=0.9, ls=ls)
    ax.set_xticks(xs, [str(c) for c in order], fontsize=8)
    ax.set_xlabel("dictionary component (top 10 by |AUC-0.5|)")
    ax.set_ylabel("affect selectivity, AUC (neg vs pos)")
    ax.set_ylim(0.14, 0.95)
    ax.set_title(f"{t3['n_components']} sparse components, "
                 f"{t3['n_pass_2species']} selective in >=2 species, "
                 f"{t3['n_pass_3species']} in all three", fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower right")
    pas = t3.get("components", [])
    if pas:
        c = pas[0]["component"]
        if c in list(order):
            i = list(order).index(c)
            ax.annotate("the only component selective in two\nspecies: cat+dog, pig at chance",
                        xy=(i - 0.27, 0.293), xytext=(i + 0.30, 0.185), fontsize=7.8,
                        arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.text(0.015, 0.985, "dashed lines = the >0.60 / <0.40 selectivity criterion\n"
            "expected passes in >=2 species under label permutation: 0.00 +- 0.00\n"
            "best single atom for pigs reaches only AUC 0.580, in 5031 calls",
            transform=ax.transAxes, fontsize=7.4, va="top",
            bbox=dict(fc="#f4f4f4", ec="#bbb", boxstyle="round,pad=0.35"))

    p = os.path.join(OUT, "shared_direction.png")
    fig.savefig(p, dpi=155)
    print("wrote", p)


if __name__ == "__main__":
    main()
