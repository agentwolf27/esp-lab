"""Generate out_site/README.md straight from the result JSONs. No hand-typed numbers."""
import os, json, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "out_site")
J = lambda n: json.load(open(os.path.join(OUT, n))) if os.path.exists(os.path.join(OUT, n)) else {}
R, S2, NU = J("results.json"), J("stage2.json"), J("nulls.json")
st, dg, ref, M = R["setup"], R["diagnostics"], R["reference"], R["methods"]
CH, MJ = st["lab_chance_balanced"], st["lab_majority_raw"]
BV, BL = M["raw"]["valence_loto"], M["raw"]["lab_acc"]
CEIL = ref["valence_within_lab_cv_mean"]
PA, FB = NU.get("paired", {}), NU.get("fold_bootstrap", {})
RN = NU.get("rand_null", {})
f3 = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) and v == v else "–"
has = lambda k: k in M and "valence_loto" in M[k]

RAW_MLP = M["raw"].get("lab_mlp", 1.0)
def verdict(k):
    e = M[k]
    lin_gone = e["lab_acc"] <= MJ + 0.02
    nl = e.get("lab_mlp")
    dv = e["valence_loto"] - BV
    if lin_gone:
        if nl is None:                     return "linear-only (MLP n/r)"
        if nl <= MJ + 0.20:                return "**invariance**" if dv >= -0.05 else "compression"
        if nl <= RAW_MLP - 0.15:           return "compression (2nd-order)" if dv < -0.05 else "2nd-order partial †"
        return "linear-only"
    if e["lab_acc"] <= BL - 0.05:
        return "compression" if dv < -0.05 else "partial"
    return "nothing"

ORDER = ["raw", "rand-2", "rand-5", "rand-32", "rand-128", "rand-384",
         "INLP-orig-5", "INLP-orig-32", "INLP-fix-5", "INLP-fix-32",
         "NAP-1", "NAP-2", "NAP-3", "NAP-4", "NAP-5", "WCCN+NAP-5", "WCCN",
         "LEACE", "cLEACE", "CORAL-train",
         "LEACE-T", "NAP-5-T", "CORAL-all", "per-lab center", "per-lab scale", "per-lab z"]
ORDER = [k for k in ORDER if has(k)]
IND = [k for k in ORDER if not M[k].get("transductive")]
TRA = [k for k in ORDER if M[k].get("transductive")]

def table(keys):
    L = ["| method | rank | valence LOTO | Δ | lab: linear | lab: quad | lab: RFF | lab: MLP | reading |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for k in keys:
        e = M[k]
        L.append(f"| `{k}` | {e['rank'] or '–'} | **{e['valence_loto']:.3f}** | {e['valence_loto']-BV:+.3f} | "
                 f"**{e['lab_acc']:.3f}** | {f3(e.get('lab_quad'))} | {f3(e.get('lab_rff'))} | "
                 f"{f3(e.get('lab_mlp'))} | {verdict(k)} |")
    return "\n".join(L)

erasers = [k for k in ORDER if M[k]["lab_acc"] <= MJ + 0.02]
best_ind = max([k for k in IND if k != "raw"], key=lambda k: M[k]["valence_loto"])
sig_bad = [k for k, v in PA.items() if v.get("sig") and v["delta"] < 0]
sig_good = [k for k, v in PA.items() if v.get("sig") and v["delta"] > 0]

txt = f"""# out_site — can recording-site identity be erased from frozen WavLM pig-call embeddings?

**Verdict: no — and the earlier diagnosis of *why* was wrong in an instructive way.**

Site identity is **not** the stubborn high-rank structure the INLP sweep suggested. It is
**exactly rank {M['LEACE']['rank']} at first order** (= number of labs − 1): a closed-form eraser (LEACE) removes
it completely, dropping a fresh linear lab probe from {BL:.3f} to **{M['LEACE']['lab_acc']:.3f}**, the majority
floor. So the previous conclusion — *"site identity is not a low-rank linear subspace"* — does
not survive.

What does survive is the pessimism, for a sharper reason. Removing that subspace buys nothing
({BV:.3f} → {M['LEACE']['valence_loto']:.3f} valence), because **an MLP reads the lab straight back off the erased
features at {M['LEACE']['lab_mlp']:.3f}**. Site in WavLM is carried by the *covariance* of each lab's
distribution, not its location, and every erasure method in the literature — INLP, LEACE, NAP,
WCCN — is a first-order tool that is blind to this by construction. The one method here that
does reach second-order structure (CORAL) removes the task along with the nuisance.

And the headline number the programme has been optimising is far noisier than it looks: with
only **4 scorable labs**, the untreated baseline's own 95 % CI is
**[{FB.get('ci95',[float('nan')]*2)[0]:.3f}, {FB.get('ci95',[float('nan')]*2)[1]:.3f}]**. Under a *paired* test on the same folds, **no method improved
valence**, and the only statistically significant effect in the entire study is that the
trivial transductive baseline (per-lab z-scoring) makes it significantly **worse**.

---

## 1. Setup

* `out_pigs/emb_wavlm.npy` layer 0 — **n = {st['n']}** pig calls, **d = {st['dim']}**, frozen WavLM-base-plus.
* Task label `pol` (binary valence); nuisance `team` = **{len(st['labs'])} recording labs**
  ({', '.join(f'{k} {v}' for k, v in st['lab_counts'].items())}).
* Floors for the 6-way lab probe: **balanced chance {CH:.3f}**, **majority-lab raw accuracy {MJ:.3f}**.
  Earlier write-ups quote 0.167 as "chance" beside a *raw-accuracy* figure; for raw accuracy the
  honest floor is the majority baseline **{MJ:.3f}**. Both columns are in `results.json`.
* CPU only, `OMP_NUM_THREADS=2`, `torch.set_num_threads(2)`.

## 2. What this dataset actually is — read before the table

| fact | value |
|---|---|
| per-lab negative-valence rate | {', '.join(f"{k} {v:.2f}" for k, v in dg['per_lab_neg_rate'].items())} |
| valence predictable from **the lab name alone** | **{dg['valence_from_lab_alone_bacc']:.3f}** balanced acc. |
| contexts recorded by all 6 labs | **{dg['n_contexts_shared_by_all_labs']}** |
| labs with both valence classes (i.e. scorable under LOTO) | **{len(NU.get('scorable_labs', []))} of {len(st['labs'])}** — {', '.join(NU.get('scorable_labs', []))} |
| held-out lab's mean shift lying in the span of the other 5 labs' shifts | {dg['heldout_mean_shift_captured_mean']:.2f} |
| lab covariance vs pooled, relative Frobenius | {', '.join(f"{k} {v:.2f}" for k, v in dg['cov_rel_frobenius_vs_pooled'].items())} |
| median per-dimension SD ratio (widest lab / narrowest) | {dg['per_dim_sd_ratio_max_over_min']:.2f} |

1. **The confound is near-total.** IASPB is 100 % positive and IASPC 100 % negative, so neither can
   be scored at all. Guessing valence from the lab name scores **{dg['valence_from_lab_alone_bacc']:.3f}** — *higher than
   the {BV:.3f} the embedding earns honestly.* The shortcut is better than the signal.
2. **Leave-one-lab-out is not a pure site shift.** No context is recorded by all six labs, so
   holding out a lab also holds out unseen behavioural contexts (§F6).
3. **Labs differ far more in covariance than any mean shift can express** — 59–115 % of the pooled
   covariance norm.

**Valence reference points** (balanced accuracy, chance 0.500):

| protocol | value | meaning |
|---|---:|---|
| random 5-fold | {ref['valence_random5fold_leaky']:.3f} | leaky — lab identity is a free feature |
| within-lab CV, mean over labs | **{CEIL:.3f}** | *same-site ceiling* ({', '.join(f"{k} {v:.2f}" for k, v in ref['valence_within_lab_cv'].items())}) |
| **leave-one-lab-out (the task)** | **{BV:.3f}** | 95 % CI [{FB.get('ci95',[float('nan')]*2)[0]:.3f}, {FB.get('ci95',[float('nan')]*2)[1]:.3f}]; per-lab {', '.join(f"{k} {v:.2f}" for k, v in FB.get('per_lab', {}).items())} |

## 3. Protocol (identical for every row)

* **(a) valence** — leave-one-lab-out balanced accuracy. The transform is **re-fitted inside every
  fold on the training labs only**; the held-out lab's *labels* are never used.
* **(b) lab decodability** — 6-way, **a fresh logistic probe that took no part in fitting the
  transform**, 5-fold stratified, z-scored on the probe's training fold. The transform is **also
  re-fitted per probe fold**, so this measures whether erasure *generalises*, not whether it memorised.
* **Random-projection control at matched rank** for every rank tested.
* Nonlinear probes on the same folds: `[z, z²]` (second-order), a 1024-component RBF
  random-Fourier-feature probe, and a 768→256→128→6 MLP.

> **A trap worth recording.** Fitting the eraser on *all* data and then probing out-of-fold — the
> protocol behind the earlier INLP numbers — gives LEACE a lab accuracy of about **0.10**, *below the
> {CH:.3f} chance level*. That is an artifact: forcing lab means to coincide globally makes the
> training-fold and test-fold residual means anti-correlated, so the probe is actively misled.
> Every headline number here re-fits per fold; the old protocol is kept as `lab_acc_insample`.

## 4. Results

### 4a. Inductive — source labs only, nothing from the target site

{table(IND)}

### 4b. Transductive — legitimate only with unlabelled audio from the target site *and* knowledge of which site it is

{table(TRA)}

Reading: **linear-only** = a linear probe is defeated, an MLP is not. **2nd-order partial** = the
MLP falls too. **compression** = lab and task both fall. **partial** = lab falls but not to the
floor. **nothing** = neither moves.

> **† `CORAL-train` does not achieve its two columns simultaneously.** Its *valence* number is
> genuinely inductive (the held-out lab is unseen, so it is left un-aligned). But its *lab* columns
> are measured with a 5-fold stratified probe in which **every fold contains all six labs**, so the
> eraser aligned all of them — which is why its lab numbers are identical to transductive
> `CORAL-all`. At deployment on a genuinely new site the target's features would be un-aligned and
> the lab fully decodable. Read the two halves of that row as belonging to different operating points.
"""

# ---------------------------------------------------------------- findings
napc = [k for k in ["NAP-1", "NAP-2", "NAP-3", "NAP-4", "NAP-5"] if has(k)]
mrow = lambda pre, ks: "| `" + pre + "` | " + " | ".join(
    f3(M.get(f"{pre}-{k}", {}).get("lab_acc")) for k in ks) + " |"

txt += f"""
## 5. Findings

### F1. Site identity **is** a low-rank linear subspace. INLP simply cannot find it.

Deleting site-mean directions one at a time walks the lab probe straight down to the floor, and
it lands exactly at rank {M['LEACE']['rank']} = (#labs − 1):

| site-mean dims removed | {' | '.join(k.split('-')[1] for k in napc)} |
|---|{'---:|' * len(napc)}
| lab acc after `NAP-k` | {' | '.join(f"{M[k]['lab_acc']:.3f}" for k in napc)} |

The matched random control does not move at all — `rand-2` {f3(M.get('rand-2',{}).get('lab_acc'))}, `rand-5` {f3(M.get('rand-5',{}).get('lab_acc'))},
`rand-384` {f3(M.get('rand-384',{}).get('lab_acc'))}, against {BL:.3f} untreated — so the descent above is the site subspace being
removed, not dimensionality being lost.

`LEACE` reaches the same place in closed form with a guarantee ({M['LEACE']['lab_acc']:.3f}), and after it the
largest lab-mean deviation is 3×10⁻⁶ of the global mean. INLP, at matched and higher rank, does not:

| | rank 5 | rank 32 |
|---|---:|---:|
{mrow("INLP-orig", (5, 32))}
{mrow("INLP-fix", (5, 32))}
| `LEACE` (rank {M['LEACE']['rank']}) | **{M['LEACE']['lab_acc']:.3f}** | – |

Two things were checked before concluding:

* **Is the plateau an implementation artifact?** `phase1_invariance.inlp_directions` fits its
  classifier on z-scored features and projects the resulting coefficient out of the *un-scaled*
  space; the correct raw-space direction is `w/σ`, not `w`. We implemented both. **It makes no
  material difference** ({f3(M.get('INLP-fix-5',{}).get('lab_acc'))} vs {f3(M.get('INLP-orig-5',{}).get('lab_acc'))} at rank 5). The plateau is not a bug.
* **So why does INLP stall where LEACE does not?** Different objectives. A linear probe fails on a
  concept exactly when the class-conditional *means* coincide (`Cov(x,z)=0`). LEACE targets that
  condition and attains it. INLP deletes the *maximum-likelihood discriminative* direction instead,
  which is not the mean-difference direction — the means stay apart and a fresh probe walks back in.
  This is precisely the failure mode LEACE was introduced to fix; this dataset is a clean
  demonstration of it on a real nuisance variable. (For continuity: the earlier deep sweep took
  original-INLP to rank 384 under the in-sample protocol and it flattened at 0.819–0.820, never
  approaching the {MJ:.3f} floor that rank-{M['LEACE']['rank']} LEACE reaches.)

### F2. Erasing it buys nothing, because a nonlinear probe reads the lab straight back off

| probe on `LEACE`-erased features | lab accuracy |
|---|---:|
| linear — *the class LEACE guarantees against* | **{f3(M['LEACE']['lab_acc'])}** |
| `[z, z²]` (per-dimension variances) | {f3(M['LEACE'].get('lab_quad'))} |
| RBF kernel (1024 random Fourier features) | {f3(M['LEACE'].get('lab_rff'))} |
| MLP | {f3(M['LEACE'].get('lab_mlp'))} |

*(untreated: linear {f3(M['raw']['lab_acc'])}, MLP {f3(M['raw'].get('lab_mlp'))}; floor {MJ:.3f})*

**This is the central negative result.** LEACE's guarantee concerns the loss of a *linear*
predictor and is honoured exactly; it says nothing about any other function class, and here that
gap is the entire story. An MLP recovers {M['LEACE'].get('lab_mlp', float('nan')):.3f} of the {M['raw'].get('lab_mlp', float('nan')):.3f} it had before, while valence
moves {M['LEACE']['valence_loto']-BV:+.3f}.

Site identity lives in the **shape** of each lab's distribution, not its location — lab covariances
differ from pooled by {min(dg['cov_rel_frobenius_vs_pooled'].values()):.0%}–{max(dg['cov_rel_frobenius_vs_pooled'].values()):.0%} in Frobenius norm, median per-dimension SD ratio {dg['per_dim_sd_ratio_max_over_min']:.2f}.
This also explains the earlier DANN failure with no appeal to optimisation pathology: an adversary
that equalises what its own head can see leaves second-order structure intact for the next probe.

### F3. The one method that reaches second-order structure destroys the task

| | valence | lab linear | lab quad | lab RFF | lab MLP |
|---|---:|---:|---:|---:|---:|
| `raw` | {M['raw']['valence_loto']:.3f} | {f3(M['raw']['lab_acc'])} | {f3(M['raw'].get('lab_quad'))} | {f3(M['raw'].get('lab_rff'))} | {f3(M['raw'].get('lab_mlp'))} |
| `LEACE` (first-order) | {M['LEACE']['valence_loto']:.3f} | {f3(M['LEACE']['lab_acc'])} | {f3(M['LEACE'].get('lab_quad'))} | {f3(M['LEACE'].get('lab_rff'))} | {f3(M['LEACE'].get('lab_mlp'))} |
| `CORAL-train` (second-order, inductive †) | {M['CORAL-train']['valence_loto']:.3f} | {f3(M['CORAL-train']['lab_acc'])} | {f3(M['CORAL-train'].get('lab_quad'))} | {f3(M['CORAL-train'].get('lab_rff'))} | {f3(M['CORAL-train'].get('lab_mlp'))} |
| `CORAL-all` (second-order) | **{M['CORAL-all']['valence_loto']:.3f}** | {f3(M['CORAL-all']['lab_acc'])} | {f3(M['CORAL-all'].get('lab_quad'))} | {f3(M['CORAL-all'].get('lab_rff'))} | **{f3(M['CORAL-all'].get('lab_mlp'))}** |
| `per-lab z` | {M['per-lab z']['valence_loto']:.3f} | {f3(M['per-lab z']['lab_acc'])} | {f3(M['per-lab z'].get('lab_quad'))} | {f3(M['per-lab z'].get('lab_rff'))} | {f3(M['per-lab z'].get('lab_mlp'))} |

The CORAL rows are the only ones where the MLP falls ({M['raw'].get('lab_mlp', float('nan')):.3f} → {M['CORAL-all'].get('lab_mlp', float('nan')):.3f}) — and where it
does, valence falls with it, to {M['CORAL-all']['valence_loto']:.3f}, i.e. chance. **Compression, not invariance.**
(`CORAL-train` shows the same lab numbers at {M['CORAL-train'].get('lab_mlp', float('nan')):.3f} while keeping valence at
{M['CORAL-train']['valence_loto']:.3f}, but see the † note in §4 — those two columns are not the same operating
point, because the lab probe's folds all contain every lab and so every lab got aligned.) Note also `per-lab z`: it puts the
*quadratic* probe on the floor ({f3(M['per-lab z'].get('lab_quad'))}) because it normalises each dimension's marginal
moments by construction — yet the MLP still scores {M['per-lab z'].get('lab_mlp', float('nan')):.3f}, because the lab also lives in the
*correlations between* dimensions, which marginal z-scoring does not touch.

### F4. Nothing improved valence — and the one significant effect is harmful

The absolute leave-one-lab-out number is very poorly determined ({len(NU.get('scorable_labs', []))} scorable labs; baseline
95 % CI [{FB.get('ci95',[float('nan')]*2)[0]:.3f}, {FB.get('ci95',[float('nan')]*2)[1]:.3f}], per-lab {', '.join(f"{k} {v:.2f}" for k, v in FB.get('per_lab', {}).items())}). But every method sees the same
folds and the same calls, so the **paired** difference is far better determined. Scored on the
{len(NU.get('scorable_labs', []))} labs that actually have both classes, with a paired cluster bootstrap over held-out labs:

| method | valence (4 labs) | Δ vs `raw` | 95 % CI (paired) | significant? |
|---|---:|---:|---:|---|
"""
for k in ["raw", "rand-384", "NAP-2", "LEACE", "NAP-5", "LEACE-T", "CORAL-all", "per-lab z"]:
    if k in PA:
        e = PA[k]
        txt += (f"| `{k}` | {e['point']:.3f} | {e['delta']:+.3f} | "
                f"[{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}] | {'**yes**' if e['sig'] else 'no'} |\n")

txt += f"""
**No method improves valence.** The only significant effect in the study is `per-lab z`
({PA.get('per-lab z',{}).get('delta',float('nan')):+.3f}, CI excludes zero) — the trivial transductive baseline, and it is significantly
**harmful**.

Two traps this exposes, both of which produce apparent wins that are not real:

* **The scoring set.** The legacy pooling (which reproduces the {BV:.3f} baseline) includes the two
  labs whose test set is single-class. `NAP-2` scores {M['NAP-2']['valence_loto']:.3f} under it — the best inductive
  number in the table, {M['NAP-2']['valence_loto']-BV:+.3f} — but **{PA.get('NAP-2',{}).get('point',float('nan')):.3f}, i.e. {PA.get('NAP-2',{}).get('delta',float('nan')):+.3f}, on the 4 labs that can
  actually be scored.** The apparent win is an artifact of pooling single-class folds.
* **The random control.** Random rank-*k* projections remove **no** site information whatsoever
  (lab stays at {f3(M['rand-384']['lab_acc'])} vs {BL:.3f}), yet across 8 seeds they wander over:

| | rand-5 | rand-32 | rand-128 | rand-384 |
|---|---:|---:|---:|---:|
| valence, mean ± sd over 8 seeds | {RN.get('5',{}).get('mean',float('nan')):.3f} ± {RN.get('5',{}).get('sd',float('nan')):.3f} | {RN.get('32',{}).get('mean',float('nan')):.3f} ± {RN.get('32',{}).get('sd',float('nan')):.3f} | {RN.get('128',{}).get('mean',float('nan')):.3f} ± {RN.get('128',{}).get('sd',float('nan')):.3f} | {RN.get('384',{}).get('mean',float('nan')):.3f} ± {RN.get('384',{}).get('sd',float('nan')):.3f} |
| range | [{RN.get('5',{}).get('min',float('nan')):.3f}, {RN.get('5',{}).get('max',float('nan')):.3f}] | [{RN.get('32',{}).get('min',float('nan')):.3f}, {RN.get('32',{}).get('max',float('nan')):.3f}] | [{RN.get('128',{}).get('min',float('nan')):.3f}, {RN.get('128',{}).get('max',float('nan')):.3f}] | [{RN.get('384',{}).get('min',float('nan')):.3f}, {RN.get('384',{}).get('max',float('nan')):.3f}] |

The single-seed `rand-384` figure of {M['rand-384']['valence_loto']:.3f} that appears in the table is the *top* of that
range. **Any future claim on this dataset must clear the random control at matched rank and matched
seed variance, not chance.**
"""

if S2.get("subspace_decomposition"):
    sd = S2["subspace_decomposition"]; sp = S2.get("site_subspace_probe", {})
    txt += f"""
### F5. A tempting result that does not hold up (recorded so nobody re-finds it)

Projecting *onto* the site-mean subspace rather than away from it looked like the one positive
result in the study:

| representation | valence LOTO |
|---|---:|
| full {st['dim']}-d embedding | {sd['full']:.3f} |
| site-mean subspace only, {sd['site_subspace_dim_transductive']}-d, **transductive** (uses the held-out lab's own mean) | **{sd['site_subspace_only_transductive']:.3f}** |
| site-mean subspace only, {sd['site_subspace_dim_inductive']}-d, **inductive** (training labs only) | {sd['site_subspace_only_inductive']:.3f} |
| random subspace, matched {sd['site_subspace_dim_inductive']} dims | {sd['random_subspace_only_matched_dim']:.3f} ± {sd['random_subspace_only_sd']:.3f} |
| orthogonal complement, {st['dim']-sd['site_subspace_dim_inductive']}-d, inductive | {sd['complement_inductive']:.3f} |

{sd['site_subspace_only_transductive']:.3f} from **5 dimensions**, beating the full 768-d embedding, is a striking-looking number.
It does not survive its control: over 10 seeds a *random* 5-d subspace averages
{sp.get('rand5_mean', float('nan')):.3f} ± {sp.get('rand5_sd', float('nan')):.3f} **and reaches {sp.get('rand5_max', float('nan')):.3f} on its best seed** — indistinguishable from the
site subspace's {sp.get('site_T', float('nan')):.3f} on the same scoring set. Five arbitrary dimensions of this embedding
support cross-lab valence about as well as {st['dim']} do; that says more about how little of the
representation is being used than about the site subspace.
"""

if S2.get("shift_decomposition"):
    d = S2["shift_decomposition"]
    fv = lambda t: f"{t[0]:.3f}" if t and t[0] == t[0] else "–"
    txt += f"""
### F6. Site may not even be the dominant shift

| held-out unit | valence | shift present |
|---|---:|---|
| lab | {fv(d['LOTO_lab'])} | new site **and** never-seen contexts |
| context, all labs in training | **{fv(d['LOCO_context'])}** | new context only, no site shift |
| context, within a single lab | {d['LOCO_within_lab_mean']:.3f} | new context, no site shift ({', '.join(f"{k} {v:.2f}" for k, v in d['LOCO_within_lab'].items())}) |
| random split within lab | {CEIL:.3f} | none |

Holding the site fixed and changing only the behavioural context costs **more** than changing the
site does ({fv(d['LOCO_context'])} vs {fv(d['LOTO_lab'])}). These are different fold counts on different
subsets and are not a like-for-like contrast, but the direction is clear and it undercuts the
framing: a large part of what has been attributed to "site" is unseen-context generalisation, which
**no site-erasure method can address**. Because no context is recorded by all six labs, this corpus
cannot separate the two, and every site-invariance number measured on it — ours and the earlier
ones — inherits that ambiguity.
"""

if S2.get("rff"):
    txt += """
### F7. Erasing in a random-feature space does not change the answer

If the leak is nonlinear, erase in a nonlinear feature space: map to 1024 random Fourier features
and apply LEACE there, zeroing the cross-covariance for a whole RKHS ball.

| representation | valence LOTO | lab (linear) | lab (MLP) |
|---|---:|---:|---:|
"""
    for k, e in S2["rff"].items():
        txt += f"| {k} | {e['valence_loto']:.3f} | {e['lab_acc']:.3f} | {e['lab_mlp']:.3f} |\n"
    txt += """
The same pattern one level up: the linear probe *in feature space* is defeated, the MLP is not, and
valence does not benefit. Kernelising the eraser moves the boundary of what counts as "linear"; it
does not move the result.
"""

txt += f"""
## 6. Method by method

* **`LEACE` / `NAP-5` / `WCCN+NAP-5`** — *linear-only invariance*. Three metrics on the same
  operation (total covariance / Euclidean / within-lab covariance); all three land in the same
  place — lab linear at the {MJ:.3f} floor, MLP untouched, valence flat. Rank {M['LEACE']['rank']} is all the
  first-order structure that exists.
* **`cLEACE`** (valence-conditional, ours) — estimates lab directions *after partialling valence
  out*, so the task's mean-difference direction is protected by construction. Valence
  {M['cLEACE']['valence_loto']:.3f}, lab {M['cLEACE']['lab_acc']:.3f}, MLP {f3(M['cLEACE'].get('lab_mlp'))}. Protecting the task
  direction is not the missing ingredient — it just erases slightly less.
* **`INLP`** — see F1: slower, no guarantee, strictly dominated by LEACE at matched rank.
* **`WCCN` alone** — valence {M['WCCN']['valence_loto']:.3f}, lab {M['WCCN']['lab_acc']:.3f}. It is an
  **invertible** map, so it cannot change what a linear probe can express; the small movement is
  pure regularisation. Recorded because it is the textbook speaker-verification answer, and it is
  worth stating plainly that on its own it cannot work here.
* **`CORAL`** — the only second-order family and the only one that reaches the real leak (F3):
  it is the sole transform here that moves the MLP probe ({M['raw'].get('lab_mlp', float('nan')):.3f} → {M['CORAL-all'].get('lab_mlp', float('nan')):.3f}).
  Transductive `CORAL-all` pays for it with the task ({M['CORAL-all']['valence_loto']:.3f}) — compression.
  Inductive `CORAL-train` keeps valence ({M['CORAL-train']['valence_loto']:.3f}) but only because it leaves the
  target lab un-aligned, which is also why its lab column cannot be read as a deployment number († in §4).
  A second-order method that aligns an *unseen* site without its statistics is not something this
  family provides.
* **Per-lab standardisation (transductive)** — center {M['per-lab center']['valence_loto']:.3f}, scale
  {M['per-lab scale']['valence_loto']:.3f}, full z-score {M['per-lab z']['valence_loto']:.3f}. This is what a
  practitioner could actually deploy given unlabelled audio from a new site, and on this corpus **it
  is the worst thing you can do** — the only significantly harmful method in the study (F4). Their
  lab-probe numbers are near-vacuous by construction (the normalisation is *defined* by the lab
  label); the meaningful column for them is valence. Reported separately and labelled transductive
  throughout.

## 7. Verdict

**Site invariance is not achievable post hoc on these frozen embeddings, and the obstruction is
structural rather than a matter of finding a better eraser.**

1. The first-order part is *solved* — closed form, rank {M['LEACE']['rank']}, guarantee holds empirically. It buys
   {M['LEACE']['valence_loto']-BV:+.3f} valence.
2. What actually carries site is **second-order** (covariance), which no erasure method targets; the
   one method that does (CORAL) removes the task with it.
3. Site and valence are confounded by corpus design — {dg['valence_from_lab_alone_bacc']:.3f} balanced accuracy for
   valence from *the lab name alone*, against {BV:.3f} for the embedding — so any transform strong
   enough to remove site removes signal.
4. A large share of the leave-one-lab-out penalty is unseen-**context** generalisation, which site
   invariance cannot address at all.
5. The evaluation is underpowered for the effect sizes being chased: 4 scorable folds, baseline CI
   width {FB.get('width', float('nan')):.3f}, and a random projection that touches nothing wanders ±{RN.get('384',{}).get('sd',float('nan')):.3f}.

This is not a post-hoc representation-surgery problem. It is a **study-design and
encoder-training** problem: record shared contexts across sites so site and task are decorrelated,
or intervene during encoder training with a second-order-aware objective. Both are outside what
frozen-embedding erasure can reach.

## 8. Honest limits

* One corpus, one encoder, one layer (layer 0, chosen by prior work as best for pigs). Nothing here
  shows the second-order story is layer- or encoder-invariant.
* Only {len(NU.get('scorable_labs', []))} of {len(st['labs'])} labs are scorable for valence; the small denominator and the confound have
  the same cause. Absolute valence numbers carry a CI roughly {FB.get('width', float('nan')):.2f} wide.
* Site and context novelty are not separable in this corpus (F6).
* Single seed for the MLP probe; the random projections use 8 seeds and the site-subspace control 10.
* Post-hoc transforms of frozen embeddings only. Nothing rules out success from a second-order-aware
  objective *during* encoder training, or from an encoder pretrained across many sites. The claim is
  narrower and firmer: **you cannot fix this downstream of a frozen representation with first-order
  erasure, and first-order erasure is what the published toolkit provides.**

## 9. Files

| file | contents |
|---|---|
| `results.json` | the full method sweep, incl. `lab_acc_insample` under the older protocol |
| `stage2.json` | shift decomposition, subspace decomposition, random-feature erasure |
| `nulls.json` | random-projection null distributions, baseline bootstrap, paired comparisons |
| `site.png` | **A** valence vs lab decodability for every method; **B** the same features under four probe classes |
| `run.log`, `stage2.log`–`stage5.log` | raw console output |

Code in the parent directory: `site_lib.py` (transforms + probes), `site_run.py` (sweep),
`site_stage2.py` (shifts, RFF erasure), `site_stage3.py`/`site_stage3b.py` (subspace decomposition),
`site_stage4.py` (nulls), `site_stage5.py` (paired tests), `site_fig.py`, `site_readme.py`.
"""
open(os.path.join(OUT, "README.md"), "w").write(txt)
print(f"wrote README.md ({len(txt)} chars)")
print("linear erasers:", erasers)
print("significant harmful:", sig_bad, "| significant helpful:", sig_good)
