"""Figure for the multiband experiment -> out_multiband/multiband.png"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "out_multiband")
res = json.load(open(os.path.join(OUT, "results.json")))

CY, AM, MG, GY = "#0A626D", "#C87A12", "#8A3459", "#6C8084"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

CONDS = [("baseband", "baseband\n(0-8 kHz)", GY),
         ("timeexp4x", "time-expansion\n4x", CY),
         ("multiband_mean", "multiband\nmean-fuse", AM),
         ("multiband_concat", "multiband\nconcat-fuse", MG)]
ENCS = [e for e in ["wavlm", "aves-bio"] if f"{e}_baseband" in res]
ENCL = {"wavlm": "WavLM-base-plus", "aves-bio": "AVES-bio"}

fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), dpi=160,
                         gridspec_kw={"width_ratios": [1.15, 1.0, 0.95]})

# ---- A: best accuracy per condition, grouped by encoder
ax = axes[0]
xp = np.arange(len(CONDS)); w = 0.8 / len(ENCS)
for j, e in enumerate(ENCS):
    vals = [res[f"{e}_{c}"]["best"] for c, _, _ in CONDS]
    off = (j - (len(ENCS) - 1) / 2) * w
    bars = ax.bar(xp + off, vals, w * 0.92,
                  color=[c for _, _, c in CONDS], alpha=1.0 if j == 0 else 0.5,
                  edgecolor="white", linewidth=0.6,
                  hatch=None if j == 0 else "///")
    for x, v in zip(xp + off, vals):
        ax.text(x, v + .008, f"{v:.3f}", ha="center", fontsize=7)
ax.axhline(0.10, ls="--", c="k", lw=1, alpha=.6)
ax.text(len(CONDS) - .55, 0.115, "chance", fontsize=7.5)
ax.set_xticks(xp); ax.set_xticklabels([l for _, l, _ in CONDS], fontsize=7.5)
ax.set_ylabel("identity accuracy (best layer)"); ax.set_ylim(0, 0.80)
h = [plt.Rectangle((0, 0), 1, 1, fc="0.35", alpha=1.0 if j == 0 else 0.5,
                   hatch=None if j == 0 else "///") for j in range(len(ENCS))]
ax.legend(h, [ENCL[e] for e in ENCS], fontsize=7.5, frameon=False, loc="upper left")
ax.set_title("A · 10-way bat identity, 1,000 calls", fontsize=10)

# ---- B: per-layer curves
ax = axes[1]
mk = {"wavlm": "o-", "aves-bio": "s--"}
for e in ENCS:
    for c, lab, col in CONDS:
        v = res[f"{e}_{c}"]["per_layer"]
        ax.plot(range(len(v)), v, mk[e], c=col, ms=2.6, lw=1.2,
                alpha=1.0 if e == "wavlm" else 0.55)
ax.set_xlabel("encoder layer"); ax.set_ylabel("identity accuracy")
ax.set_title("B · per layer (solid WavLM, dashed AVES)", fontsize=10)
hh = [plt.Line2D([], [], color=col, lw=2) for _, _, col in CONDS]
ax.legend(hh, [l.replace("\n", " ") for _, l, _ in CONDS], fontsize=7, frameon=False,
          loc="upper right")

# ---- C: single-band accuracy -> where does identity actually live?
ax = axes[2]
nb = sum(1 for k in res if k.startswith(f"{ENCS[0]}_multiband_band"))
xp = np.arange(nb); w = 0.8 / len(ENCS)
for j, e in enumerate(ENCS):
    vals = [res[f"{e}_multiband_band{k}_only"]["best"] for k in range(nb)]
    off = (j - (len(ENCS) - 1) / 2) * w
    ax.bar(xp + off, vals, w * 0.92, color=AM, alpha=1.0 if j == 0 else 0.5,
           edgecolor="white", linewidth=0.6, hatch=None if j == 0 else "///")
    for x, v in zip(xp + off, vals):
        ax.text(x, v + .008, f"{v:.2f}", ha="center", fontsize=7)
for e, ls in zip(ENCS, ["-", "--"]):
    ax.axhline(res[f"{e}_timeexp4x"]["best"], ls=ls, c=CY, lw=1.1, alpha=.85)
ax.text(nb - 0.5, res[f"{ENCS[0]}_timeexp4x"]["best"] + .012, "time-expansion 4x",
        fontsize=7, ha="right", color=CY)
ax.axhline(0.10, ls="--", c="k", lw=1, alpha=.6)
ax.set_xticks(xp)
ax.set_xticklabels([f"band {k}\n{8*k}-{8*(k+1)} kHz" for k in range(nb)], fontsize=7)
ax.set_ylabel("identity accuracy"); ax.set_ylim(0, 0.80)
ax.set_title("C · one band at a time", fontsize=10)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "multiband.png"))
print("wrote", os.path.join(OUT, "multiband.png"))
