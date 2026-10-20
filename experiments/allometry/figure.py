from __future__ import annotations
import json, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(OUT, "results.json")))
d = pd.read_csv(os.path.join(OUT, "analysis_set.csv"))
C = {"cat": "#E4761B", "dog": "#2C6FA6", "pig": "#3F8F5E"}
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": .8, "figure.dpi": 150})

fig = plt.figure(figsize=(13.0, 9.8))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.3, .8, 1.0],
                      hspace=.50, wspace=.36, top=.875, bottom=.10, left=.055, right=.985)

# ------------------------------------------------------------------ (a) law
ax = fig.add_subplot(gs[0, :2])
law = R["allometry"]["group_level_all"]
rng = np.random.default_rng(1)
for s in ["cat", "dog", "pig"]:
    q = d[d.species == s]
    ax.scatter(q.log_mass + rng.normal(0, .013, len(q)), q.log_f0, s=4, alpha=.11,
               color=C[s], lw=0, zorder=1)
gg = pd.DataFrame(law["groups"])
for _, g in gg.iterrows():
    ax.scatter(g.log_mass, g.log_f0, s=26 + 3.0 * np.sqrt(g.n), color=C[g.species],
               edgecolor="white", lw=1.1, zorder=4)
x = np.linspace(d.log_mass.min() - .10, d.log_mass.max() + .10, 50)
mid = law["intercept"] + law["slope"] * np.mean(x)
for nm, b, ls in [("isometric theory  $b$=-0.333", -1 / 3, (0, (5, 2))),
                  ("Bowling'17 carnivores  $b$=-0.335", -.3347, (0, (1.2, 1.6))),
                  ("Bowling'17 primates  $b$=-0.819", -.8187, (0, (6, 2, 1.4, 2)))]:
    ax.plot(x, mid + b * (x - np.mean(x)), ls=ls, color="#5A5A5A", lw=1.15, zorder=3, label=nm)
ax.plot(x, law["intercept"] + law["slope"] * x, "k-", lw=2.2, zorder=5,
        label=f"our fit  $b$={law['slope']:+.3f}   $R^2$={law['r2']:.2f}   p={law['p']:.2f}  (n.s.)")
for _, g in gg[gg.species == "pig"].iterrows():
    ax.annotate(g.mass_group.replace("pig_", "pig "), (g.log_mass, g.log_f0),
                textcoords="offset points",
                xytext=((-6, 12) if "Piglet" in g.mass_group else (9, -3)), fontsize=7,
                color="#26603C", fontweight="bold",
                ha=("right" if "Piglet" in g.mass_group else "left"))
ax.annotate("4 cat breed×sex groups\n(imputed mass, 4–8 kg)", (np.log10(4.4), 2.60),
            fontsize=6.8, color="#A9520F", ha="center")
ax.annotate("10 dogs, measured mass", (np.log10(26), 3.12), fontsize=6.8, color="#1D5A88", ha="center")
h, l = ax.get_legend_handles_labels()
ax.legend([h[-1]] + h[:-1], [l[-1]] + l[:-1], fontsize=6.9, loc="lower left",
          frameon=False, borderaxespad=.2, labelspacing=.35)
ax.set_xlabel("log$_{10}$ body mass (kg)"); ax.set_ylabel("log$_{10}$ $F_0$ (Hz)")
ax.set_ylim(1.55, 3.42)
ax.set_title("(a)  Acoustic allometry across three species: our own data do not establish the law",
             loc="left", fontweight="bold", fontsize=10, pad=8)
ax.text(.995, .975, "the two pig points that break the fit are a call-type artefact —\n"
                    "sampled piglets mostly scream, sampled weaners mostly grunt",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color="#666", style="italic")

# --------------------------------------------------------------- (b) in-dog
ax = fig.add_subplot(gs[0, 2])
w = R["allometry"]["within_dog_group"]
gd = pd.DataFrame(w["groups"])
q = d[d.species == "dog"]
ax.scatter(q.log_mass, q.log_f0, s=5, alpha=.15, color=C["dog"], lw=0)
ax.scatter(gd.log_mass, gd.log_f0, s=44, color=C["dog"], edgecolor="white", lw=1.2, zorder=4)
xx = np.linspace(gd.log_mass.min() - .06, gd.log_mass.max() + .06, 30)
ax.plot(xx, np.mean(gd.log_f0) - (1 / 3) * (xx - np.mean(gd.log_mass)),
        color="#C8102E", lw=6, alpha=.35, solid_capstyle="round", zorder=5,
        label="isometric $-1/3$ prediction")
ax.plot(xx, w["intercept"] + w["slope"] * xx, "k-", lw=1.8, zorder=6,
        label=f"our fit  $b$={w['slope']:+.3f}")
ax.legend(fontsize=7, frameon=False, loc="upper right", borderaxespad=.2)
ax.set_xlabel("log$_{10}$ body mass (kg)"); ax.set_ylabel("log$_{10}$ $F_0$ (Hz)")
ax.set_title("(b)  Within dogs, real masses:\n      here the law does hold", loc="left",
             fontweight="bold", fontsize=10, pad=8)
ax.text(.03, .04, f"$R^2$={w['r2']:.2f},  p={w['p']:.3f}\nthe fitted and theoretical\nslopes all but coincide",
        transform=ax.transAxes, fontsize=7, va="bottom")

# ---------------------------------------------------------- (c) transfer
ax = fig.add_subplot(gs[1, :2])
tab = R["transfer_with_nopitch_baseline"]
pairs = list(tab.keys())
VAR = ["nopitch", "raw", "allo_all", "allo_loso", "zspec", "center"]
LBL = {"nopitch": "no pitch at all (duration+energy)", "raw": "(a) raw $F_0$",
       "allo_all": "(b) allometric residual", "allo_loso": "(b′) allometric, law fitted excl. test species",
       "zspec": "(c) within-species $z$", "center": "species-mean centred"}
COLV = {"nopitch": "#9AA0A6", "raw": "#B23A48", "allo_all": "#127A63",
        "allo_loso": "#5FB49C", "zspec": "#3C6FB0", "center": "#8E7CC3"}
wd, xs = .137, np.arange(len(pairs))
for i, v in enumerate(VAR):
    ax.bar(xs + (i - 2.5) * wd, [tab[p][v] for p in pairs], wd * .9, color=COLV[v], lw=0)
for j, p in enumerate(pairs):
    n95 = np.mean([R["transfer"]["A_pitch_isolated"][v][p]["null_p95"]
                   for v in VAR if v in R["transfer"]["A_pitch_isolated"]])
    ax.hlines(n95, xs[j] - 3.15 * wd, xs[j] + 3.15 * wd, color="k", lw=1.5, zorder=6)
ax.axhline(.5, color="#999", ls="--", lw=1)
ax.set_xticks(xs); ax.set_xticklabels([p.replace("->", " → ") for p in pairs], fontsize=9)
ax.set_ylim(.44, .79); ax.set_ylabel("balanced accuracy")
ax.set_title("(c)  Cross-species affect transfer: every pitch normalisation lands on the no-pitch baseline",
             loc="left", fontweight="bold", fontsize=10, pad=26)
ax.legend(handles=[Patch(facecolor=COLV[v], label=LBL[v]) for v in VAR], fontsize=6.8,
          ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.005),
          columnspacing=1.4, handlelength=1.2, handleheight=.9)
m = R["transfer_means_over_6_pairs"]
SHORT = {"nopitch": "no pitch", "raw": "(a) raw", "allo_all": "(b) allometric",
         "allo_loso": "(b\u2032) allometric LOSO", "zspec": "(c) within-sp. $z$",
         "center": "sp.-mean centred"}
ax.text(.5, .978, "mean over all 6 directed pairs:   " +
        "   ".join(f"{SHORT[k]} {m[k]:.3f}" for k in VAR),
        transform=ax.transAxes, ha="center", va="top", fontsize=6.6,
        bbox=dict(fc="#F1F1F1", ec="none", pad=3.5))
ax.text(.5, -.185, "black bar = 95th percentile of the within-individual/lab label-permutation null "
        "(200 draws) — it sits far above 0.5, so naive chance badly overstates significance",
        transform=ax.transAxes, ha="center", fontsize=6.9, color="#444", style="italic")

# --------------------------------------------------------- (d) mechanism
ax = fig.add_subplot(gs[1, 2])
why = R["why_probe_coefficients_negative_is_class1"]
cues = ["pitch", "dur_call", "energy"]
xs = np.arange(3)
for i, s in enumerate(["cat", "dog", "pig"]):
    ax.bar(xs + (i - 1) * .26, [why[s][c] for c in cues], .24, color=C[s], label=s, lw=0)
ax.axhline(0, color="k", lw=.8)
ax.set_xticks(xs); ax.set_xticklabels(["pitch", "call duration", "energy"])
ax.tick_params(axis="x", pad=1)
ax.set_ylabel("probe weight   (+ = predicts NEGATIVE affect)")
ax.set_ylim(-1.75, 2.15)
ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper center")
ax.set_title("(d)  Why it fails", loc="left", fontweight="bold", fontsize=10, pad=8)
cs = R["affect_axis_cosine_between_species"]
ax.text(.5, -.155, "Pitch points the opposite way in pigs than in cats and dogs.\n"
        "Only call duration agrees in all three.\nAffect-axis cosines:  " +
        ",  ".join(f"{k} {v:+.2f}" for k, v in cs.items()) +
        "\nA location/scale fix cannot repair a sign flip.",
        transform=ax.transAxes, ha="center", va="top", fontsize=7, color="#333")

fig.suptitle("Does normalising pitch by body size make animal calls comparable across species?     No.",
             fontsize=13.5, fontweight="bold", y=.975)
fig.text(.5, .945, "cat meows (n=333) · dog barks (n=233) · pig calls (n=1260) · "
         "NEGATIVE = cat isolation / dog aggression / pig Neg valence",
         ha="center", fontsize=8, color="#555")
fig.savefig(os.path.join(OUT, "allometry.png"), dpi=165, bbox_inches="tight", facecolor="white")
print("wrote allometry.png")
