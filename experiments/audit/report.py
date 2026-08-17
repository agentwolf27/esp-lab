"""Assemble out_audit/results.json, README.md and the three figures."""
from __future__ import annotations

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CY, AM, MG, GY = "#0A626D", "#C87A12", "#8A3459", "#6C8084"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

P = json.load(open(os.path.join(HERE, "points.json")))
LAW = json.load(open(os.path.join(HERE, "law.json")))
REC = json.load(open(os.path.join(HERE, "recovery.json")))

DS = {
    "cats":  {"label": "CatMeows", "task": "3-class context (brushing / food / isolation)",
              "n": 440, "group": "cat", "n_groups": 21, "chance": 1 / 3},
    "dogs":  {"label": "Dog barks", "task": "3-class context (contact / play / aggression)",
              "n": 693, "group": "dog", "n_groups": 10, "chance": 1 / 3},
    "pigs":  {"label": "Soundwel pigs", "task": "binary valence",
              "n": 5031, "group": "lab", "n_groups": 6, "chance": 0.5},
    "bats":  {"label": "BEANS bats (exploratory)", "task": "4-class context (sleep/fight/isolation/feed)",
              "n": 378, "group": "emitter", "n_groups": 10, "chance": 0.25},
}
FEAT = {"wavlm": "WavLM-base-plus", "hubert": "HuBERT-base", "wav2vec2": "wav2vec2-base",
        "aves-bio": "AVES-bio", "mfcc": "MFCC-85", "egemaps": "eGeMAPS-88",
        "acoustic18": "paper's 18 features", "duration": "log-duration only"}
ORDER = ["wavlm", "hubert", "wav2vec2", "aves-bio", "egemaps", "acoustic18", "mfcc", "duration"]


def rows_for(ds, include_variants=False):
    out = []
    feats = [f for f in ORDER if any(p["dataset"] == ds and p["features"] == f
                                     and not p.get("variant") for p in P["points"])]
    for f in feats:
        pts = [p for p in P["points"] if p["dataset"] == ds and p["features"] == f
               and not p.get("variant")]
        best = max(pts, key=lambda p: p["heldout"])
        out.append({"dataset": ds, "features": f, "n_layers": len(pts), "best": best,
                    "mean_heldout": float(np.mean([p["heldout"] for p in pts])),
                    "mean_random": float(np.mean([p["random"] for p in pts])),
                    "mean_identity": float(np.mean([p["identity"] for p in pts])),
                    "mean_inflation": float(np.mean([p["inflation"] for p in pts]))})
    return out


TABLE = [r for ds in ["cats", "dogs", "pigs", "bats"] for r in rows_for(ds)]


# ------------------------------------------------------------------ markdown
def md_table():
    L = ["| Dataset | Features | Held-out group | Random | Inflation | Group identity | Chance (task / identity) |",
         "|---|---|---|---|---|---|---|"]
    for r in TABLE:
        d = DS[r["dataset"]]
        b = r["best"]
        lay = f" (L{b['layer']})" if r["n_layers"] > 1 else ""
        floor = " *" if r["features"] == "duration" else ""
        L.append(f"| {d['label']} | {FEAT[r['features']]}{lay}{floor} | **{b['heldout']:.3f}** | "
                 f"{b['random']:.3f} | +{b['inflation']:.3f} | {b['identity']:.3f} | "
                 f"{d['chance']:.3f} / {1/d['n_groups']:.3f} |")
    return "\n".join(L)


def md_mean_table():
    L = ["| Dataset | Encoder | Layers | Held-out (mean) | Random (mean) | Inflation (mean) | Identity (mean) |",
         "|---|---|---|---|---|---|---|"]
    for r in TABLE:
        if r["n_layers"] < 2:
            continue
        L.append(f"| {DS[r['dataset']]['label']} | {FEAT[r['features']]} | {r['n_layers']} | "
                 f"{r['mean_heldout']:.3f} | {r['mean_random']:.3f} | +{r['mean_inflation']:.3f} | "
                 f"{r['mean_identity']:.3f} |")
    return "\n".join(L)


# ------------------------------------------------------------------ fig 1
def fig_table():
    rows = [r for r in TABLE if r["dataset"] != "bats" and r["features"] != "duration"]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.3), dpi=160,
                             gridspec_kw={"width_ratios": [6, 3, 3]})
    for ax, ds in zip(axes, ["cats", "dogs", "pigs"]):
        rr = [r for r in rows if r["dataset"] == ds]
        names = [FEAT[r["features"]] for r in rr]
        yp = np.arange(len(rr))[::-1]; h = 0.26
        ho = [r["best"]["heldout"] for r in rr]
        rd = [r["best"]["random"] for r in rr]
        idn = [r["best"]["identity"] for r in rr]
        ax.barh(yp + h, rd, h, color=MG, alpha=.55, label="situation, random split (leaky)")
        ax.barh(yp, ho, h, color=CY, label="situation, held-out group (honest)")
        ax.barh(yp - h, idn, h, color=AM, alpha=.85, label="group identity")
        ax.axvline(DS[ds]["chance"], ls="--", c="k", lw=1, alpha=.6)
        ax.axvline(1 / DS[ds]["n_groups"], ls=":", c=AM, lw=1, alpha=.8)
        for i, (a, b) in zip(yp, zip(ho, rd)):
            ax.text(b + .015, i + h, f"{b:.2f}", va="center", fontsize=7, color=MG)
            ax.text(a + .015, i, f"{a:.2f}", va="center", fontsize=7, color=CY, fontweight="bold")
        ax.set_yticks(yp); ax.set_yticklabels(names, fontsize=8); ax.set_xlim(0, 1.06)
        ax.set_title(f"{DS[ds]['label']}\n{DS[ds]['n']} clips · {DS[ds]['n_groups']} "
                     f"{DS[ds]['group']}s · chance {DS[ds]['chance']:.2f}", fontsize=8)
        ax.set_xlabel("accuracy (situation = balanced)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, ncol=3, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Frozen encoders encode WHO/WHERE more reliably than WHAT-SITUATION "
                 "(best layer per feature set)", fontsize=10.5)
    fig.text(0.5, 0.058, "dashed = task chance · dotted amber = identity chance",
             ha="center", fontsize=7.2, color=GY)
    fig.tight_layout(rect=(0, 0.075, 1, 0.93))
    fig.savefig(os.path.join(HERE, "audit_table.png")); plt.close(fig)


# ------------------------------------------------------------------ fig 2
def fig_law():
    from scipy import stats
    col = {"cats": CY, "dogs": AM, "pigs": MG, "bats": GY}
    mk = {"wavlm": "o", "hubert": "s", "wav2vec2": "^", "aves-bio": "D",
          "egemaps": "P", "mfcc": "X", "acoustic18": "*"}
    pts = [p for p in P["points"] if not p.get("variant") and not p.get("floor")]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), dpi=160)

    ax = axes[0]
    prim = [p for p in pts if p["dataset"] != "bats"]
    x = np.array([p["identity"] for p in prim]); y = np.array([p["inflation"] for p in prim])
    for p in prim:
        ax.scatter(p["identity"], p["inflation"], s=34, c=col[p["dataset"]],
                   marker=mk.get(p["features"], "o"), alpha=.8, edgecolors="none")
    sl, ic, r, pv, se = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, sl * xs + ic, c="k", lw=1.4, alpha=.75)
    b = LAW["primary_cats_dogs_pigs"]
    ax.set_title("pooled over datasets, feature sets and layers", fontsize=9)
    ax.text(.03, .97, f"n={b['pooled']['n']}\nPearson r={b['pooled']['pearson_r']:.3f}, "
                      f"p={b['pooled']['pearson_p']:.1e}\n"
                      f"Spearman rho={b['pooled']['spearman_rho']:.3f}, "
                      f"p={b['pooled']['spearman_p']:.1e}\n"
                      f"within-dataset partial r={b['within_dataset_partial']['r']:.3f}, "
                      f"p={b['within_dataset_partial']['p']:.1e}\n"
                      f"cluster level (n={b['cluster_level']['n']}) r="
                      f"{b['cluster_level']['pearson_r']:.3f}, p={b['cluster_level']['pearson_p']:.2f}",
            transform=ax.transAxes, va="top", fontsize=7.4,
            bbox=dict(fc="white", ec=GY, lw=.6, alpha=.9))
    for d in ["cats", "dogs", "pigs"]:
        ax.scatter([], [], c=col[d], s=34, label=f"{DS[d]['label']} "
                                                 f"(n={sum(1 for p in prim if p['dataset']==d)})")
    ax.legend(fontsize=7.4, loc="lower right", frameon=False)

    ax = axes[1]
    for d in ["cats", "dogs", "pigs"]:
        dd = [p for p in prim if p["dataset"] == d]
        xx = np.array([p["identity"] for p in dd]); yy = np.array([p["inflation"] for p in dd])
        ax.scatter(xx - xx.mean(), yy - yy.mean(), s=34, c=col[d], alpha=.8, edgecolors="none",
                   label=f"{DS[d]['label']}  r={b['per_dataset'][d]['pearson_r']:.2f}, "
                         f"p={b['per_dataset'][d]['pearson_p']:.3f}")
    xa = np.concatenate([np.array([p["identity"] for p in prim if p["dataset"] == d]) -
                         np.mean([p["identity"] for p in prim if p["dataset"] == d])
                         for d in ["cats", "dogs", "pigs"]])
    ya = np.concatenate([np.array([p["inflation"] for p in prim if p["dataset"] == d]) -
                         np.mean([p["inflation"] for p in prim if p["dataset"] == d])
                         for d in ["cats", "dogs", "pigs"]])
    sl2, ic2, *_ = stats.linregress(xa, ya)
    xs = np.linspace(xa.min(), xa.max(), 50)
    ax.plot(xs, sl2 * xs + ic2, c="k", lw=1.4, alpha=.75)
    ax.axhline(0, c=GY, lw=.6); ax.axvline(0, c=GY, lw=.6)
    ax.set_title("same points, each dataset centred on its own mean", fontsize=9)
    ax.legend(fontsize=7.4, loc="lower right", frameon=False)
    ax.set_xlabel("group-identity accuracy (centred)")

    axes[0].set_xlabel("group-identity accuracy")
    axes[0].set_ylabel("inflation  (random − held-out group)")
    axes[1].set_ylabel("inflation (centred)")
    fig.suptitle("Does identity decodability predict evaluation inflation?", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(HERE, "inflation_law.png")); plt.close(fig)


# ------------------------------------------------------------------ fig 3
def fig_recovery():
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), dpi=160)
    spec = [("cats", axes[0], [("cats_wavlm", CY, "WavLM L3"), ("cats_egemaps", AM, "eGeMAPS")]),
            ("pigs", axes[1], [("pigs_wavlm", CY, "WavLM L0"), ("pigs_egemaps", AM, "eGeMAPS")])]
    for ds, ax, series in spec:
        for key, c, lab in series:
            r = REC[key]
            ks = [p["eff_k"] for p in r["curve"]]
            mu = np.array([p["mean"] for p in r["curve"]])
            sd = np.array([p["sd"] for p in r["curve"]])
            ax.plot(ks, mu, "o-", c=c, label=lab)
            ax.fill_between(ks, mu - sd, mu + sd, color=c, alpha=.2, lw=0)
            ax.axhline(r["random_ceiling"], ls="--", c=c, lw=1, alpha=.65)
            ax.text(ks[-1], r["random_ceiling"] + .004,
                    f"random-split ceiling {r['random_ceiling']:.3f}", fontsize=6.8,
                    color=c, ha="right")
        ax.axhline(DS[ds]["chance"], ls=":", c="k", lw=1, alpha=.6)
        ax.set_xlabel("labelled clips added from the held-out group (effective k)")
        ax.set_ylabel("balanced accuracy on the held-out group's test half")
        ax.legend(fontsize=7.6, frameon=False, loc="lower right")
        note = ("cats saturate at k≈11: the median cat has only ~19 clips,\n"
                "so there is nothing left to add" if ds == "cats"
                else "labs are large, so k is genuinely 100")
        ax.set_title(f"{DS[ds]['label']} — {note}", fontsize=8.5)
    fig.suptitle("How much labelled target-group data repairs the shift?", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(HERE, "recovery.png")); plt.close(fig)


# ------------------------------------------------------------------ results.json
def write_results():
    out = {
        "what": "Multi-dataset audit: do frozen audio encoders encode WHO/WHERE more than "
                "WHAT-SITUATION, and does group-identity decodability predict how much a "
                "random split inflates the reported score?",
        "protocol": {
            "situation_metric": "balanced accuracy (chance = 1/n_classes)",
            "identity_metric": "plain accuracy, random stratified 5-fold "
                               "(chance = 1/n_groups; group sizes are unequal)",
            "heldout": "leave-one-group-out, predictions pooled over folds",
            "random": "stratified 5-fold, shuffle, seed 0",
            "estimators": {
                "cat_style": "per-fold StandardScaler + LogisticRegression(C=1, max_iter=3000); "
                             "cats, dogs, bats",
                "pig_style": "global z-score + LogisticRegression(C=1, class_weight=balanced); "
                             "pigs; LOTO scored only on the 4 labs holding both valences"},
            "audio": "mono, 16 kHz, 6 s truncation, per-clip mean/std norm, batch size 1",
            "encoder": "microsoft/wavlm-base-plus, frozen, hidden states mean-pooled over time",
        },
        "datasets": DS,
        "table": TABLE,
        "all_points": P["points"],
        "reproduction_checks": P["checks"],
        "inflation_law": LAW,
        "recovery": REC,
        "verdict": "Supported WITHIN a dataset, not established ACROSS datasets. Identity "
                   "decodability predicts inflation among feature sets and layers of the same "
                   "dataset (within-dataset partial r=0.646, df=79, p=7.1e-11; within-encoder, "
                   "layer-only r=0.636, df=70, p=2.0e-9; all six encoders positive), but the "
                   "conservative cluster-level test over the 12 independent (dataset x feature "
                   "set) units is null (r=0.221, p=0.489), so 'more identity leakage => more "
                   "inflation' is a within-corpus selection rule, not a cross-corpus law.",
        "caveats": [
            "Layer points from one encoder are not independent; the pooled p-values are "
            "anti-conservative. The cluster-level test (one point per dataset x feature set) "
            "is the conservative version and it is not significant (n=12, r=0.221, p=0.489).",
            "Identity also correlates with the honest held-out score (r=0.509, p=8.8e-07), not "
            "only with the leaky one (r=0.754, p=1.9e-16). Part of the relation is therefore "
            "'better features decode everything better', not pure leakage; the leakage account "
            "survives only because the leaky number moves more.",
            "Bats are exploratory only: no context label is decodable at all (held-out 0.25-0.30 "
            "against a 0.25 chance), so their near-zero inflation is a degenerate zero, not "
            "support for the law. Including bats raises the pooled r to 0.874 purely as a "
            "leverage effect; the primary fit excludes them.",
            "Dog 'context' is confounded with recording session, and dogs contribute only 10 "
            "groups; the leave-one-dog-out number is not a leave-one-session-out number.",
            "Cat and pig headline rows are re-used from the earlier runs (out_enc, out_pigs). "
            "Pig WavLM layer 0 was recomputed here as a check and matched to <0.002.",
            "Recovery uses a fixed stratified test half per group so k=0 and k>0 are scored on "
            "exactly the same clips; k is capped at the adaptation pool size.",
        ],
    }
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    return out


def write_readme():
    b = LAW["primary_cats_dogs_pigs"]; bb = LAW["with_bats"]
    rec_lines = []
    for key, lab in [("cats_wavlm", "CatMeows · WavLM L3"), ("cats_egemaps", "CatMeows · eGeMAPS"),
                     ("pigs_wavlm", "Soundwel pigs · WavLM L0"), ("pigs_egemaps", "Soundwel pigs · eGeMAPS")]:
        r = REC[key]
        cells = " | ".join(f"{p['mean']:.3f}" for p in r["curve"])
        eff = "/".join(f"{p['eff_k']:.0f}" for p in r["curve"])
        rec_lines.append(f"| {lab} | {cells} | {r['random_ceiling']:.3f} | {eff} |")
    rec_tbl = ("| Dataset · features | k=0 | k=10 | k=25 | k=50 | k=100 | random-split ceiling | "
               "effective k |\n|---|---|---|---|---|---|---|---|\n" +
               "\n".join(rec_lines) +
               "\n\nAccuracies are the mean over 10 (cats) / 5 (pigs) resamples of which clips "
               "are added; sd is in `recovery.json` and is ≤0.016 everywhere.")

    txt = f"""# Multi-dataset audit — identity leakage and evaluation inflation

Three species, four datasets, frozen encoders, one protocol. For every
(dataset, feature set, layer) we measure the same three numbers:

* **held-out group** — situation accuracy under leave-one-group-out (the honest number)
* **random** — situation accuracy under a random stratified 5-fold (the leaky number)
* **group identity** — how well the *same* features decode which animal / which lab

and define **inflation = random − held-out group**.

Groups are: cat (21), dog (10), recording lab (6), bat emitter (10). Situation
scores are balanced accuracy, chance = 1/n_classes. Identity is plain accuracy,
chance = 1/n_groups (group sizes are unequal, so the majority-group rate is
higher than chance; it is recorded in `results.json`).

## Audit table (best layer per feature set)

{md_table()}

`*` log-duration only, included as a floor: dog contexts are **not** simply
different recording lengths.

Mean over all layers, for the multi-layer encoders:

{md_mean_table()}

## The inflation law

Pooling every (dataset, feature set, layer) point, excluding the exploratory
bats and the duration floor:

* **pooled** n={b['pooled']['n']}, Pearson r={b['pooled']['pearson_r']:.3f} (p={b['pooled']['pearson_p']:.2e}),
  Spearman rho={b['pooled']['spearman_rho']:.3f} (p={b['pooled']['spearman_p']:.2e})
* **within-dataset partial** (each dataset centred on its own mean)
  r={b['within_dataset_partial']['r']:.3f}, df={b['within_dataset_partial']['df']}, p={b['within_dataset_partial']['p']:.2e}
* **per dataset**: """ + "; ".join(
        f"{d} r={v['pearson_r']:.3f} (n={v['n']}, p="
        f"{v['pearson_p']:.3f})" if v["pearson_p"] >= 1e-3 else
        f"{d} r={v['pearson_r']:.3f} (n={v['n']}, p={v['pearson_p']:.2e})"
        for d, v in b["per_dataset"].items()) + f"""
* **cluster level**, one point per (dataset × feature set):
  n={b['cluster_level']['n']}, r={b['cluster_level']['pearson_r']:.3f} (p={b['cluster_level']['pearson_p']:.3f}),
  rho={b['cluster_level']['spearman_rho']:.3f} (p={b['cluster_level']['spearman_p']:.3f}) — **not significant**
* **neural points only** (768-d, so the effect is not "handcrafted vs deep"):
  n={b['neural_only']['n']}, r={b['neural_only']['pearson_r']:.3f} (p={b['neural_only']['pearson_p']:.2e});
  within-dataset r={b['neural_only_within_dataset']['r']:.3f} (p={b['neural_only_within_dataset']['p']:.2e})
* **cats, one point per feature set** (the n=6 successor to the earlier n=5 test that gave
  r=0.81, p=0.094): r={b['cats_encoder_level']['pearson_r']:.3f}, p={b['cats_encoder_level']['pearson_p']:.3f}
* **within one encoder** — centred inside each (dataset × feature set) cluster, so the only
  variation left is layer-to-layer inside a single frozen model:
  n={b['within_encoder_partial']['n']}, {b['within_encoder_partial']['n_clusters']} clusters,
  df={b['within_encoder_partial']['df']}, r={b['within_encoder_partial']['r']:.3f},
  p={b['within_encoder_partial']['p']:.2e}. Per encoder: """ + "; ".join(
        f"{k} r={v['pearson_r']:+.2f} (p="
        + (f"{v['pearson_p']:.3f})" if v["pearson_p"] >= 1e-3 else f"{v['pearson_p']:.1e})")
        for k, v in b["per_encoder"].items()) + f""" — all six positive
  (sign test p=0.031), three individually significant.
* **leverage**: dropping the most extreme point ({b['drop_highest_leverage']['dropped']}) *raises*
  the pooled r to {b['drop_highest_leverage']['pearson_r']:.3f} (p={b['drop_highest_leverage']['pearson_p']:.2e});
  jackknifing out each feature set in turn keeps r in
  [{b['jackknife_drop_one_featureset']['min_r']:.3f}, {b['jackknife_drop_one_featureset']['max_r']:.3f}].

In `inflation_law.png` colour is the dataset and marker shape is the feature set
(circle WavLM, square HuBERT, triangle wav2vec2, diamond AVES-bio, plus eGeMAPS,
cross MFCC-85, star the pigs' published 18 features).

Decomposition: identity tracks the **leaky** number
(r={b['identity_vs_random']['pearson_r']:.3f}, p={b['identity_vs_random']['pearson_p']:.2e}) more
strongly than the honest one
(r={b['identity_vs_heldout']['pearson_r']:.3f}, p={b['identity_vs_heldout']['pearson_p']:.2e}),
which is the direction the leakage account predicts.

Including the exploratory bats lifts the pooled correlation to
r={bb['pooled']['pearson_r']:.3f} (n={bb['pooled']['n']}, p={bb['pooled']['pearson_p']:.2e}), but that is a
leverage artefact: within bats the correlation is *negative*
(r={bb['per_dataset']['bats']['pearson_r']:.3f}, p={bb['per_dataset']['bats']['pearson_p']:.3f}) and no bat
context label is decodable at all, so their inflation is a degenerate zero.

## How much labelled target-group data repairs it

For each group a fixed stratified **test half** is set aside once; k labelled clips are drawn
from the other half and added to the training set, so k=0 and k>0 are scored on exactly the
same clips. k is capped at the pool size — the median cat only has ~19 clips in total.

{rec_tbl}

Cats: ~11 clips from the target cat is enough to reach the random-split ceiling
({REC['cats_wavlm']['curve'][0]['mean']:.3f} → {REC['cats_wavlm']['curve'][-1]['mean']:.3f} vs a
{REC['cats_wavlm']['random_ceiling']:.3f} ceiling), i.e. the whole "shift" is target-individual
idiosyncrasy that a handful of labels absorbs. Pigs: 100 clips from the target lab closes only
{100*(REC['pigs_wavlm']['curve'][-1]['mean']-REC['pigs_wavlm']['curve'][0]['mean'])/(REC['pigs_wavlm']['random_ceiling']-REC['pigs_wavlm']['curve'][0]['mean']):.0f}%
of the gap and is still climbing — lab shift is deeper than individual shift.
eGeMAPS recovers far less in both, i.e. the encoder's advantage is partly that it is
*adaptable*, not that it is *transferable*.

## Verdict

{write_results()['verdict']}

## Files

`results.json` (everything), `audit_table.png`, `inflation_law.png`, `recovery.png`,
`points.json` (per-layer points), `law.json`, `recovery.json`.
Scripts: `embed_dogs.py`, `egemaps_extra.py`, `audit.py`, `report.py`.
Caches: `emb_dogs_wavlm.npy` (13×693×768), `egemaps_{{cats,dogs,pigs}}.npy`, `mfcc_dogs.npy`.

## Caveats

""" + "\n".join(f"* {c}" for c in write_results()["caveats"]) + "\n"
    open(os.path.join(HERE, "README.md"), "w").write(txt)


if __name__ == "__main__":
    fig_table(); fig_law(); fig_recovery()
    write_results(); write_readme()
    print(md_table())
    print()
    print("wrote results.json, README.md, audit_table.png, inflation_law.png, recovery.png")
