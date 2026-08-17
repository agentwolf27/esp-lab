"""Figures for the overnight run. Writes PNGs into out_fig/."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(D, "out_fig"); os.makedirs(FIG, exist_ok=True)
enc = json.load(open(os.path.join(D, "out_enc", "results.json")))
val = json.load(open(os.path.join(D, "out_cross", "validation.json")))
rep = json.load(open(os.path.join(D, "out_cross", "replication.json")))

CY, AM, MG, GY = "#0A626D", "#C87A12", "#8A3459", "#6C8084"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------- 1. encoder comparison
order = sorted(enc.items(), key=lambda kv: kv[1]["best"]["ctx_loco"])
names = [k for k, _ in order]
ctx = [v["best"]["ctx_loco"] for _, v in order]
leak = [v["best"]["ctx_rand"] for _, v in order]
ident = [v["best"]["identity"] for _, v in order]
fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=160)
yp = np.arange(len(names)); h = 0.26
ax.barh(yp + h, leak, h, color=MG, alpha=.55, label="context, random split (leaky)")
ax.barh(yp, ctx, h, color=CY, label="context, held-out animal (honest)")
ax.barh(yp - h, ident, h, color=AM, alpha=.8, label="individual identity")
ax.axvline(1/3, ls="--", c="k", lw=1, alpha=.6)
ax.text(1/3, len(names)-0.35, " chance (context)", fontsize=7.5, va="top")
for i, (a, b) in enumerate(zip(ctx, leak)):
    ax.text(b + .012, i + h, f"{b:.2f}", va="center", fontsize=7.5, color=MG)
    ax.text(a + .012, i, f"{a:.2f}", va="center", fontsize=7.5, color=CY, fontweight="bold")
ax.set_yticks(yp); ax.set_yticklabels(names); ax.set_xlim(0, 1)
ax.set_xlabel("balanced accuracy"); ax.legend(fontsize=7.5, loc="lower right", frameon=False)
ax.set_title("CatMeows · 3 contexts · frozen encoders, best layer each", fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIG}/encoders.png"); plt.close(fig)

# ---------------------------------------------------- 2. layer sweep, wavlm ctx vs identity
w = enc["wavlm"]["layers"]
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=160)
L = [r["layer"] for r in w]
ax.plot(L, [r["ctx_loco"] for r in w], "o-", c=CY, label="context (held-out cat)")
ax.plot(L, [r["identity"] for r in w], "s-", c=AM, label="individual identity")
ax.plot(L, [r["ctx_rand"] for r in w], "^--", c=MG, alpha=.6, label="context (random split)")
ax.axhline(1/3, ls="--", c="k", lw=1, alpha=.5)
ax.set_xlabel("WavLM layer"); ax.set_ylabel("balanced accuracy"); ax.set_ylim(0.25, .9)
ax.legend(fontsize=7.5, frameon=False)
ax.set_title("identity fades with depth; context does not", fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIG}/layers.png"); plt.close(fig)

# ---------------------------------------------------- 3. transfer per layer w/ null band
lay = val["layers"]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), dpi=160, sharey=True)
for ax, key, pk, nm, nsd, ttl in [
    (axes[0], "cat_to_dog", "p_cat_to_dog", "null_cd_mean", "null_cd_sd", "cat → dog"),
    (axes[1], "dog_to_cat", "p_dog_to_cat", "null_dc_mean", "null_dc_sd", "dog → cat")]:
    x = [r["layer"] for r in lay]; v = [r[key] for r in lay]
    mu = np.array([r[nm] for r in lay]); sd = np.array([r[nsd] for r in lay])
    ax.fill_between(x, mu - 2*sd, mu + 2*sd, color=GY, alpha=.22, label="null ±2 SD (300 perms)")
    ax.plot(x, mu, c=GY, lw=1, ls="--")
    sig = [r[pk] < .05 for r in lay]
    ax.plot(x, v, "-", c=CY, lw=1.6)
    ax.scatter([xi for xi, s in zip(x, sig) if not s], [vi for vi, s in zip(v, sig) if not s],
               c=CY, s=28, zorder=3)
    ax.scatter([xi for xi, s in zip(x, sig) if s], [vi for vi, s in zip(v, sig) if s],
               c=MG, s=60, zorder=4, marker="*", label="p < 0.05")
    ax.axhline(.5, ls=":", c="k", lw=1)
    ax.set_xlabel("WavLM layer"); ax.set_title(ttl, fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
axes[0].set_ylabel("balanced accuracy (transfer)")
fig.suptitle("affect direction learned on one species, applied to the other", fontsize=10.5)
fig.tight_layout(); fig.savefig(f"{FIG}/transfer.png"); plt.close(fig)

# ---------------------------------------------------- 4. the control that matters
fig, ax = plt.subplots(figsize=(6.6, 3.2), dpi=160)
feats = ["log_dur", "log_energy", "centroid", "zcr"]
catv = [0.28, 0.69, 0.12, 0.14]; dogv = [0.31, -0.19, -0.47, -0.48]
xp = np.arange(len(feats)); w_ = 0.36
ax.bar(xp - w_/2, catv, w_, color=CY, label="cat  (isolation − brushing)")
ax.bar(xp + w_/2, dogv, w_, color=AM, label="dog  (aggression − play)")
ax.axhline(0, c="k", lw=1)
ax.set_xticks(xp); ax.set_xticklabels(feats); ax.set_ylabel("separation (SD)")
ax.legend(fontsize=7.5, frameon=False)
ax.set_title("simple acoustics point OPPOSITE ways in the two species", fontsize=10)
for i, (a, b) in enumerate(zip(catv, dogv)):
    if a * b < 0:
        ax.text(i, max(a, b) + .06, "✗", ha="center", color=MG, fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{FIG}/acoustics.png"); plt.close(fig)

# ---------------------------------------------------- 5. replication
fig, ax = plt.subplots(figsize=(6.6, 3.0), dpi=160)
ms = list(rep.keys()); xp = np.arange(len(ms)); w_ = 0.36
dc = [rep[m]["mean_dog_to_cat"] for m in ms]; cd = [rep[m]["mean_cat_to_dog"] for m in ms]
nd = [rep[m]["null_dc_mean"] for m in ms]
b1 = ax.bar(xp - w_/2, dc, w_, color=CY, label="dog → cat")
ax.bar(xp + w_/2, cd, w_, color=GY, alpha=.65, label="cat → dog")
ax.plot(xp - w_/2, nd, "k_", ms=18, label="permutation null")
ax.axhline(.5, ls=":", c="k", lw=1)
for i, m in enumerate(ms):
    p = rep[m]["p_dog_to_cat"]
    ax.text(i - w_/2, dc[i] + .006, f"p={p:.3f}" + ("*" if p < .05 else ""),
            ha="center", fontsize=7.5, color=MG if p < .05 else GY, fontweight="bold" if p < .05 else "normal")
ax.set_xticks(xp); ax.set_xticklabels(ms); ax.set_ylim(.45, .63)
ax.set_ylabel("mean over all layers"); ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper center")
ax.set_title("selection-free replication across three speech encoders", fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIG}/replication.png"); plt.close(fig)

import base64
for f in os.listdir(FIG):
    if f.endswith(".png"):
        open(f"{FIG}/{f}.b64", "w").write(base64.b64encode(open(f"{FIG}/{f}", "rb").read()).decode())
print("figures:", sorted(x for x in os.listdir(FIG) if x.endswith(".png")))

# ---------------------------------------------------- 6. room control
room = json.load(open(os.path.join(D, "out_cross", "room_control.json")))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.2), dpi=160)
lab = ["cat→dog", "dog→cat"]
bg = [room["background_transfer"]["cat_to_dog"], room["background_transfer"]["dog_to_cat"]]
full = [rep["wavlm"]["mean_cat_to_dog"], rep["wavlm"]["mean_dog_to_cat"]]
xp = np.arange(2); w_ = .36
a1.bar(xp - w_/2, full, w_, color=CY, label="full clip (WavLM)")
a1.bar(xp + w_/2, bg, w_, color=GY, alpha=.7, label="background only")
a1.axhline(.5, ls=":", c="k", lw=1); a1.set_xticks(xp); a1.set_xticklabels(lab)
a1.set_ylim(.45, .62); a1.set_ylabel("transfer accuracy"); a1.legend(fontsize=7.5, frameon=False)
a1.set_title("the room does NOT transfer", fontsize=10)
w2 = [room["background_within_cat"], room["background_within_dog"]]
a2.bar([0, 1], w2, .5, color=MG, alpha=.8)
a2.axhline(.5, ls=":", c="k", lw=1)
a2.axhline(.79, ls="--", c=CY, lw=1.2)
a2.text(1.45, .795, "full embedding\n(cat, best)", fontsize=7, color=CY, va="bottom", ha="right")
a2.set_xticks([0, 1]); a2.set_xticklabels(["cat", "dog"]); a2.set_ylim(.45, .85)
a2.set_ylabel("within-species accuracy")
a2.set_title("…but it DOES predict context within species", fontsize=10)
for i, v in enumerate(w2):
    a2.text(i, v + .008, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{FIG}/room.png"); plt.close(fig)
import base64 as _b
open(f"{FIG}/room.png.b64","w").write(_b.b64encode(open(f"{FIG}/room.png","rb").read()).decode())
print("added room.png")
