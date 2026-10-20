"""site.png -- valence vs lab-decodability for each method, and probe-class sensitivity."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "out_site")
J = lambda n: json.load(open(os.path.join(OUT, n))) if os.path.exists(os.path.join(OUT, n)) else {}
R, S2, NU = J("results.json"), J("stage2.json"), J("nulls.json")
M = {k: v for k, v in R["methods"].items() if "valence_loto" in v}
st = R["setup"]; CH, MJ = st["lab_chance_balanced"], st["lab_majority_raw"]
BASE_V, BASE_L = M["raw"]["valence_loto"], M["raw"]["lab_acc"]
CEIL = R["reference"]["valence_within_lab_cv_mean"]
CI = NU.get("fold_bootstrap", {}).get("ci95")

SHOW = [k for k in ["raw", "rand-384", "INLP-fix-32", "NAP-2", "NAP-5", "LEACE", "cLEACE",
                    "WCCN", "CORAL-train", "LEACE-T", "CORAL-all", "per-lab scale",
                    "per-lab z"] if k in M]
FAM = lambda k: ("raw" if k == "raw" else "rand" if k.startswith("rand") else
                 "inlp" if k.startswith("INLP") else "leace" if "LEACE" in k else
                 "perlab" if k.startswith("per-lab") else "class")
COL = {"raw": "#111111", "rand": "#8a8a8a", "inlp": "#e07b39", "leace": "#2f6fbd",
       "class": "#7b4fa8", "perlab": "#1a9e6b"}

fig = plt.figure(figsize=(16.0, 6.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.52, 1.0], wspace=0.18,
                      left=.052, right=.988, top=.815, bottom=.145)

# ================================================================ A: the trade-off plane
ax = fig.add_subplot(gs[0, 0])
YLO, YHI = 0.465, 0.875
XLO, XHI = 0.02, 1.98
GUT = 1.06
if CI:
    ax.axhspan(CI[0], CI[1], color="#9aa0a6", alpha=.15, zorder=0)
ax.axhline(BASE_V, color="#444", lw=1.1, zorder=1)
ax.axhline(CEIL, color="#1a9e6b", lw=1.1, ls=":", zorder=1)
ax.axhline(0.5, color="#bbb", lw=.9, zorder=1)
ax.axvline(MJ, color="#c0392b", lw=1.1, ls="--", zorder=1)
ax.axvspan(0.80, 0.84, color="#d94f8a", alpha=.13, zorder=0)

for k in SHOW:
    v = M[k]; c = COL[FAM(k)]; T = v.get("transductive", False)
    ax.scatter(v["lab_acc"], v["valence_loto"], s=132 if k == "raw" else 88,
               c="white" if T else c, edgecolors=c, linewidths=2.1,
               marker="s" if T else "o", zorder=6)

# all labels in one gutter OUTSIDE the [0,1] metric range, evenly spaced, leader-lined
ax.axvline(1.0, color="#ccc", lw=.9, zorder=1)
grp = sorted(SHOW, key=lambda k: -M[k]["valence_loto"])
span = (YHI - .028) - (YLO + .028)
ys = [(YHI - .028) - i * span / max(len(grp) - 1, 1) for i in range(len(grp))]
for k, ly in zip(grp, ys):
    v = M[k]; c = COL[FAM(k)]; T = v.get("transductive", False)
    ax.annotate(f"{k}{'  (T)' if T else ''}      {v['valence_loto']:.3f}  /  {v['lab_acc']:.3f}",
                xy=(v["lab_acc"], v["valence_loto"]), xytext=(GUT, ly),
                textcoords="data", ha="left", va="center", fontsize=9.2,
                color="#1a1a1a", zorder=4,
                arrowprops=dict(arrowstyle="-", lw=.75, color=c, alpha=.5,
                                shrinkA=2, shrinkB=5))
ax.text(GUT, YHI - .005, "method            valence / lab", fontsize=8.4,
        color="#666", ha="left", va="center", style="italic")

ax.set_xlim(XLO, XHI); ax.set_ylim(YLO, YHI)
ax.set_xticks(np.arange(0.2, 1.01, 0.2))
ax.set_xlabel("lab decodability   (6-way, fresh linear probe, eraser re-fit out of fold)", fontsize=10)
ax.set_ylabel("valence, leave-one-lab-out (balanced acc.)", fontsize=10)
ax.set_title("A.  No method buys site-invariance and valence at once", fontsize=11.5, loc="left", pad=44)
ax.text(MJ + .016, YHI - .010, "majority-lab floor 0.291", fontsize=8, color="#c0392b",
        rotation=90, va="top", ha="left")
ax.text(0.82, YLO + .012, "INLP plateau", fontsize=8, color="#a8306a", ha="center")
ax.text(0.035, CEIL + .007, f"same-site ceiling {CEIL:.2f}", fontsize=8.2, color="#1a9e6b")
ax.text(0.035, BASE_V + .007, f"untreated baseline {BASE_V:.3f}", fontsize=8.2, color="#444")
ax.text(0.035, 0.478, "valence chance 0.500", fontsize=8.2, color="#999")
if CI:
    ax.text(0.035, CI[1] - .007, "95% CI of the untreated baseline\n(cluster bootstrap, 4 scorable labs)",
            fontsize=8, color="#666", va="top")
ax.grid(alpha=.16, lw=.6); ax.set_axisbelow(True)
ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="#2f6fbd", mec="#2f6fbd", ms=8,
                          label="inductive — source labs only"),
                   Line2D([], [], marker="s", ls="", mfc="white", mec="#1a9e6b", mew=2, ms=8,
                          label="transductive (T) — uses target-site audio")],
          fontsize=8.6, loc="lower left", bbox_to_anchor=(0.0, 1.012), ncol=2,
          frameon=False, columnspacing=1.8, handlelength=1.4)

# ================================================================ B: probe class matters
ax2 = fig.add_subplot(gs[0, 1])
keys = [k for k in ["raw", "rand-5", "LEACE", "cLEACE", "NAP-5", "LEACE-T", "CORAL-all",
                    "per-lab z"] if k in M and "lab_mlp" in M[k]]
probes = [("lab_acc", "linear", "#2f6fbd"), ("lab_quad", "quadratic  [z, z²]", "#7b4fa8"),
          ("lab_rff", "RBF kernel (RFF)", "#e07b39"), ("lab_mlp", "MLP", "#c0392b")]
w = 0.205; xs = np.arange(len(keys))
for i, (kk, lbl, c) in enumerate(probes):
    ax2.bar(xs + (i - 1.5) * w, [M[k].get(kk, np.nan) for k in keys], w,
            label=lbl, color=c, edgecolor="white", lw=.6, zorder=3)
ax2.axhline(MJ, color="#c0392b", ls="--", lw=1.1, zorder=4)
ax2.set_xticks(xs); ax2.set_xticklabels(keys, rotation=30, ha="right", fontsize=9)
ax2.set_ylabel("lab decodability (6-way accuracy)", fontsize=10)
ax2.set_ylim(0, 1.06); ax2.set_yticks(np.arange(0, 1.01, .2))
ax2.set_title("B.  Linear erasure is invisible to a nonlinear probe", fontsize=11.5, loc="left", pad=44)
ax2.set_xlim(-0.55, len(keys) - 0.45)
h, l = ax2.get_legend_handles_labels()
h.append(Line2D([], [], color="#c0392b", ls="--", lw=1.1)); l.append("majority floor 0.291")
ax2.legend(h, l, fontsize=8.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.012),
           frameon=False, columnspacing=1.4, handlelength=1.6)
ax2.grid(axis="y", alpha=.16, lw=.6); ax2.set_axisbelow(True)

fig.suptitle("Post-hoc site invariance on frozen WavLM pig-call embeddings   "
             f"(n={st['n']}, {len(st['labs'])} recording labs, layer 0)",
             fontsize=13, x=.055, ha="left", y=.972)
fig.savefig(os.path.join(OUT, "site.png"), dpi=165)
print("wrote", os.path.join(OUT, "site.png"))
