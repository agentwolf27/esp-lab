# out_site — can recording-site identity be erased from frozen WavLM pig-call embeddings?

**Verdict: no — and the earlier diagnosis of *why* was wrong in an instructive way.**

Site identity is **not** the stubborn high-rank structure the INLP sweep suggested. It is
**exactly rank 5 at first order** (= number of labs − 1): a closed-form eraser (LEACE) removes
it completely, dropping a fresh linear lab probe from 0.941 to **0.291**, the majority
floor. So the previous conclusion — *"site identity is not a low-rank linear subspace"* — does
not survive.

What does survive is the pessimism, for a sharper reason. Removing that subspace buys nothing
(0.654 → 0.657 valence), because **an MLP reads the lab straight back off the erased
features at 0.918**. Site in WavLM is carried by the *covariance* of each lab's
distribution, not its location, and every erasure method in the literature — INLP, LEACE, NAP,
WCCN — is a first-order tool that is blind to this by construction. The one method here that
does reach second-order structure (CORAL) removes the task along with the nuisance.

And the headline number the programme has been optimising is far noisier than it looks: with
only **4 scorable labs**, the untreated baseline's own 95 % CI is
**[0.526, 0.725]**. Under a *paired* test on the same folds, **no method improved
valence**, and the only statistically significant effect in the entire study is that the
trivial transductive baseline (per-lab z-scoring) makes it significantly **worse**.

---

## 1. Setup

* `out_pigs/emb_wavlm.npy` layer 0 — **n = 5031** pig calls, **d = 768**, frozen WavLM-base-plus.
* Task label `pol` (binary valence); nuisance `team` = **6 recording labs**
  (ETHZ 354, FBN 884, IASPA 1466, IASPB 522, IASPC 424, NMBU 1381).
* Floors for the 6-way lab probe: **balanced chance 0.167**, **majority-lab raw accuracy 0.291**.
  Earlier write-ups quote 0.167 as "chance" beside a *raw-accuracy* figure; for raw accuracy the
  honest floor is the majority baseline **0.291**. Both columns are in `results.json`.
* CPU only, `OMP_NUM_THREADS=2`, `torch.set_num_threads(2)`.

## 2. What this dataset actually is — read before the table

| fact | value |
|---|---|
| per-lab negative-valence rate | ETHZ 0.32, FBN 0.87, IASPA 0.76, IASPB 0.00, IASPC 1.00, NMBU 0.42 |
| valence predictable from **the lab name alone** | **0.766** balanced acc. |
| contexts recorded by all 6 labs | **0** |
| labs with both valence classes (i.e. scorable under LOTO) | **4 of 6** — ETHZ, FBN, IASPA, NMBU |
| held-out lab's mean shift lying in the span of the other 5 labs' shifts | 0.67 |
| lab covariance vs pooled, relative Frobenius | ETHZ 0.84, FBN 0.77, IASPA 0.59, IASPB 1.15, IASPC 0.79, NMBU 0.77 |
| median per-dimension SD ratio (widest lab / narrowest) | 1.88 |

1. **The confound is near-total.** IASPB is 100 % positive and IASPC 100 % negative, so neither can
   be scored at all. Guessing valence from the lab name scores **0.766** — *higher than
   the 0.654 the embedding earns honestly.* The shortcut is better than the signal.
2. **Leave-one-lab-out is not a pure site shift.** No context is recorded by all six labs, so
   holding out a lab also holds out unseen behavioural contexts (§F6).
3. **Labs differ far more in covariance than any mean shift can express** — 59–115 % of the pooled
   covariance norm.

**Valence reference points** (balanced accuracy, chance 0.500):

| protocol | value | meaning |
|---|---:|---|
| random 5-fold | 0.883 | leaky — lab identity is a free feature |
| within-lab CV, mean over labs | **0.831** | *same-site ceiling* (ETHZ 0.79, FBN 0.75, IASPA 0.90, NMBU 0.88) |
| **leave-one-lab-out (the task)** | **0.654** | 95 % CI [0.526, 0.725]; per-lab ETHZ 0.68, FBN 0.54, IASPA 0.76, NMBU 0.51 |

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
> 0.167 chance level*. That is an artifact: forcing lab means to coincide globally makes the
> training-fold and test-fold residual means anti-correlated, so the probe is actively misled.
> Every headline number here re-fits per fold; the old protocol is kept as `lab_acc_insample`.

## 4. Results

### 4a. Inductive — source labs only, nothing from the target site

| method | rank | valence LOTO | Δ | lab: linear | lab: quad | lab: RFF | lab: MLP | reading |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `raw` | – | **0.654** | +0.000 | **0.941** | 0.946 | 0.912 | 0.949 | nothing |
| `rand-2` | 2 | **0.653** | -0.001 | **0.940** | – | – | – | nothing |
| `rand-5` | 5 | **0.654** | +0.000 | **0.940** | 0.944 | 0.913 | 0.949 | nothing |
| `rand-32` | 32 | **0.650** | -0.004 | **0.940** | – | – | – | nothing |
| `rand-128` | 128 | **0.652** | -0.002 | **0.940** | – | – | – | nothing |
| `rand-384` | 384 | **0.683** | +0.029 | **0.935** | – | – | – | nothing |
| `INLP-orig-5` | 5 | **0.665** | +0.012 | **0.925** | – | – | – | nothing |
| `INLP-orig-32` | 32 | **0.665** | +0.012 | **0.903** | – | – | – | nothing |
| `INLP-fix-5` | 5 | **0.660** | +0.006 | **0.934** | – | – | – | nothing |
| `INLP-fix-32` | 32 | **0.665** | +0.011 | **0.911** | 0.935 | 0.901 | 0.947 | nothing |
| `NAP-1` | 1 | **0.656** | +0.002 | **0.909** | – | – | – | nothing |
| `NAP-2` | 2 | **0.684** | +0.030 | **0.772** | – | – | – | partial |
| `NAP-3` | 3 | **0.634** | -0.019 | **0.640** | – | – | – | partial |
| `NAP-4` | 4 | **0.657** | +0.003 | **0.402** | – | – | – | partial |
| `NAP-5` | 5 | **0.656** | +0.002 | **0.291** | 0.890 | 0.732 | 0.919 | linear-only |
| `WCCN+NAP-5` | 5 | **0.662** | +0.008 | **0.291** | 0.906 | 0.772 | 0.915 | linear-only |
| `WCCN` | – | **0.628** | -0.026 | **0.936** | – | – | – | nothing |
| `LEACE` | 5 | **0.657** | +0.003 | **0.291** | 0.906 | 0.772 | 0.918 | linear-only |
| `cLEACE` | 5 | **0.673** | +0.019 | **0.346** | 0.909 | 0.780 | 0.928 | partial |
| `CORAL-train` | – | **0.651** | -0.003 | **0.291** | 0.334 | 0.396 | 0.694 | 2nd-order partial † |

### 4b. Transductive — legitimate only with unlabelled audio from the target site *and* knowledge of which site it is

| method | rank | valence LOTO | Δ | lab: linear | lab: quad | lab: RFF | lab: MLP | reading |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `LEACE-T` | 5 | **0.547** | -0.106 | **0.095** | 0.922 | 0.779 | 0.933 | linear-only |
| `NAP-5-T` | 5 | **0.552** | -0.102 | **0.093** | – | – | – | linear-only (MLP n/r) |
| `CORAL-all` | – | **0.516** | -0.138 | **0.291** | 0.334 | 0.396 | 0.694 | compression (2nd-order) |
| `per-lab center` | – | **0.537** | -0.116 | **0.291** | – | – | – | linear-only (MLP n/r) |
| `per-lab scale` | – | **0.630** | -0.024 | **0.967** | – | – | – | nothing |
| `per-lab z` | – | **0.506** | -0.148 | **0.291** | 0.291 | 0.740 | 0.970 | linear-only |

Reading: **linear-only** = a linear probe is defeated, an MLP is not. **2nd-order partial** = the
MLP falls too. **compression** = lab and task both fall. **partial** = lab falls but not to the
floor. **nothing** = neither moves.

> **† `CORAL-train` does not achieve its two columns simultaneously.** Its *valence* number is
> genuinely inductive (the held-out lab is unseen, so it is left un-aligned). But its *lab* columns
> are measured with a 5-fold stratified probe in which **every fold contains all six labs**, so the
> eraser aligned all of them — which is why its lab numbers are identical to transductive
> `CORAL-all`. At deployment on a genuinely new site the target's features would be un-aligned and
> the lab fully decodable. Read the two halves of that row as belonging to different operating points.

## 5. Findings

### F1. Site identity **is** a low-rank linear subspace. INLP simply cannot find it.

Deleting site-mean directions one at a time walks the lab probe straight down to the floor, and
it lands exactly at rank 5 = (#labs − 1):

| site-mean dims removed | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| lab acc after `NAP-k` | 0.909 | 0.772 | 0.640 | 0.402 | 0.291 |

The matched random control does not move at all — `rand-2` 0.940, `rand-5` 0.940,
`rand-384` 0.935, against 0.941 untreated — so the descent above is the site subspace being
removed, not dimensionality being lost.

`LEACE` reaches the same place in closed form with a guarantee (0.291), and after it the
largest lab-mean deviation is 3×10⁻⁶ of the global mean. INLP, at matched and higher rank, does not:

| | rank 5 | rank 32 |
|---|---:|---:|
| `INLP-orig` | 0.925 | 0.903 |
| `INLP-fix` | 0.934 | 0.911 |
| `LEACE` (rank 5) | **0.291** | – |

Two things were checked before concluding:

* **Is the plateau an implementation artifact?** `phase1_invariance.inlp_directions` fits its
  classifier on z-scored features and projects the resulting coefficient out of the *un-scaled*
  space; the correct raw-space direction is `w/σ`, not `w`. We implemented both. **It makes no
  material difference** (0.934 vs 0.925 at rank 5). The plateau is not a bug.
* **So why does INLP stall where LEACE does not?** Different objectives. A linear probe fails on a
  concept exactly when the class-conditional *means* coincide (`Cov(x,z)=0`). LEACE targets that
  condition and attains it. INLP deletes the *maximum-likelihood discriminative* direction instead,
  which is not the mean-difference direction — the means stay apart and a fresh probe walks back in.
  This is precisely the failure mode LEACE was introduced to fix; this dataset is a clean
  demonstration of it on a real nuisance variable. (For continuity: the earlier deep sweep took
  original-INLP to rank 384 under the in-sample protocol and it flattened at 0.819–0.820, never
  approaching the 0.291 floor that rank-5 LEACE reaches.)

### F2. Erasing it buys nothing, because a nonlinear probe reads the lab straight back off

| probe on `LEACE`-erased features | lab accuracy |
|---|---:|
| linear — *the class LEACE guarantees against* | **0.291** |
| `[z, z²]` (per-dimension variances) | 0.906 |
| RBF kernel (1024 random Fourier features) | 0.772 |
| MLP | 0.918 |

*(untreated: linear 0.941, MLP 0.949; floor 0.291)*

**This is the central negative result.** LEACE's guarantee concerns the loss of a *linear*
predictor and is honoured exactly; it says nothing about any other function class, and here that
gap is the entire story. An MLP recovers 0.918 of the 0.949 it had before, while valence
moves +0.003.

Site identity lives in the **shape** of each lab's distribution, not its location — lab covariances
differ from pooled by 59%–115% in Frobenius norm, median per-dimension SD ratio 1.88.
This also explains the earlier DANN failure with no appeal to optimisation pathology: an adversary
that equalises what its own head can see leaves second-order structure intact for the next probe.

### F3. The one method that reaches second-order structure destroys the task

| | valence | lab linear | lab quad | lab RFF | lab MLP |
|---|---:|---:|---:|---:|---:|
| `raw` | 0.654 | 0.941 | 0.946 | 0.912 | 0.949 |
| `LEACE` (first-order) | 0.657 | 0.291 | 0.906 | 0.772 | 0.918 |
| `CORAL-train` (second-order, inductive †) | 0.651 | 0.291 | 0.334 | 0.396 | 0.694 |
| `CORAL-all` (second-order) | **0.516** | 0.291 | 0.334 | 0.396 | **0.694** |
| `per-lab z` | 0.506 | 0.291 | 0.291 | 0.740 | 0.970 |

The CORAL rows are the only ones where the MLP falls (0.949 → 0.694) — and where it
does, valence falls with it, to 0.516, i.e. chance. **Compression, not invariance.**
(`CORAL-train` shows the same lab numbers at 0.694 while keeping valence at
0.651, but see the † note in §4 — those two columns are not the same operating
point, because the lab probe's folds all contain every lab and so every lab got aligned.) Note also `per-lab z`: it puts the
*quadratic* probe on the floor (0.291) because it normalises each dimension's marginal
moments by construction — yet the MLP still scores 0.970, because the lab also lives in the
*correlations between* dimensions, which marginal z-scoring does not touch.

### F4. Nothing improved valence — and the one significant effect is harmful

The absolute leave-one-lab-out number is very poorly determined (4 scorable labs; baseline
95 % CI [0.526, 0.725], per-lab ETHZ 0.68, FBN 0.54, IASPA 0.76, NMBU 0.51). But every method sees the same
folds and the same calls, so the **paired** difference is far better determined. Scored on the
4 labs that actually have both classes, with a paired cluster bootstrap over held-out labs:

| method | valence (4 labs) | Δ vs `raw` | 95 % CI (paired) | significant? |
|---|---:|---:|---:|---|
| `raw` | 0.664 | +0.000 | [+0.000, +0.000] | no |
| `rand-384` | 0.683 | +0.016 | [-0.001, +0.035] | no |
| `NAP-2` | 0.657 | -0.015 | [-0.044, +0.010] | no |
| `LEACE` | 0.631 | -0.035 | [-0.111, +0.066] | no |
| `NAP-5` | 0.629 | -0.036 | [-0.115, +0.069] | no |
| `LEACE-T` | 0.566 | -0.073 | [-0.142, +0.070] | no |
| `CORAL-all` | 0.523 | -0.116 | [-0.178, +0.011] | no |
| `per-lab z` | 0.516 | -0.123 | [-0.199, -0.056] | **yes** |

**No method improves valence.** The only significant effect in the study is `per-lab z`
(-0.123, CI excludes zero) — the trivial transductive baseline, and it is significantly
**harmful**.

Two traps this exposes, both of which produce apparent wins that are not real:

* **The scoring set.** The legacy pooling (which reproduces the 0.654 baseline) includes the two
  labs whose test set is single-class. `NAP-2` scores 0.684 under it — the best inductive
  number in the table, +0.030 — but **0.657, i.e. -0.015, on the 4 labs that can
  actually be scored.** The apparent win is an artifact of pooling single-class folds.
* **The random control.** Random rank-*k* projections remove **no** site information whatsoever
  (lab stays at 0.935 vs 0.941), yet across 8 seeds they wander over:

| | rand-5 | rand-32 | rand-128 | rand-384 |
|---|---:|---:|---:|---:|
| valence, mean ± sd over 8 seeds | 0.654 ± 0.001 | 0.653 ± 0.003 | 0.656 ± 0.009 | 0.661 ± 0.013 |
| range | [0.651, 0.654] | [0.649, 0.660] | [0.647, 0.679] | [0.644, 0.683] |

The single-seed `rand-384` figure of 0.683 that appears in the table is the *top* of that
range. **Any future claim on this dataset must clear the random control at matched rank and matched
seed variance, not chance.**

### F5. A tempting result that does not hold up (recorded so nobody re-finds it)

Projecting *onto* the site-mean subspace rather than away from it looked like the one positive
result in the study:

| representation | valence LOTO |
|---|---:|
| full 768-d embedding | 0.654 |
| site-mean subspace only, 5-d, **transductive** (uses the held-out lab's own mean) | **0.718** |
| site-mean subspace only, 4-d, **inductive** (training labs only) | 0.607 |
| random subspace, matched 4 dims | 0.529 ± 0.055 |
| orthogonal complement, 764-d, inductive | 0.657 |

0.718 from **5 dimensions**, beating the full 768-d embedding, is a striking-looking number.
It does not survive its control: over 10 seeds a *random* 5-d subspace averages
0.616 ± 0.051 **and reaches 0.714 on its best seed** — indistinguishable from the
site subspace's 0.688 on the same scoring set. Five arbitrary dimensions of this embedding
support cross-lab valence about as well as 768 do; that says more about how little of the
representation is being used than about the site subspace.

### F6. Site may not even be the dominant shift

| held-out unit | valence | shift present |
|---|---:|---|
| lab | 0.654 | new site **and** never-seen contexts |
| context, all labs in training | **0.569** | new context only, no site shift |
| context, within a single lab | 0.693 | new context, no site shift (FBN 0.81, IASPA 0.67, NMBU 0.60) |
| random split within lab | 0.831 | none |

Holding the site fixed and changing only the behavioural context costs **more** than changing the
site does (0.569 vs 0.654). These are different fold counts on different
subsets and are not a like-for-like contrast, but the direction is clear and it undercuts the
framing: a large part of what has been attributed to "site" is unseen-context generalisation, which
**no site-erasure method can address**. Because no context is recorded by all six labs, this corpus
cannot separate the two, and every site-invariance number measured on it — ours and the earlier
ones — inherits that ambiguity.

### F7. Erasing in a random-feature space does not change the answer

If the leak is nonlinear, erase in a nonlinear feature space: map to 1024 random Fourier features
and apply LEACE there, zeroing the cross-covariance for a whole RKHS ball.

| representation | valence LOTO | lab (linear) | lab (MLP) |
|---|---:|---:|---:|
| RFF (no erasure) | 0.614 | 0.924 | 0.935 |
| RFF + LEACE | 0.565 | 0.291 | 0.901 |
| RFF + LEACE-T | 0.515 | 0.058 | 0.885 |

The same pattern one level up: the linear probe *in feature space* is defeated, the MLP is not, and
valence does not benefit. Kernelising the eraser moves the boundary of what counts as "linear"; it
does not move the result.

## 6. Method by method

* **`LEACE` / `NAP-5` / `WCCN+NAP-5`** — *linear-only invariance*. Three metrics on the same
  operation (total covariance / Euclidean / within-lab covariance); all three land in the same
  place — lab linear at the 0.291 floor, MLP untouched, valence flat. Rank 5 is all the
  first-order structure that exists.
* **`cLEACE`** (valence-conditional, ours) — estimates lab directions *after partialling valence
  out*, so the task's mean-difference direction is protected by construction. Valence
  0.673, lab 0.346, MLP 0.928. Protecting the task
  direction is not the missing ingredient — it just erases slightly less.
* **`INLP`** — see F1: slower, no guarantee, strictly dominated by LEACE at matched rank.
* **`WCCN` alone** — valence 0.628, lab 0.936. It is an
  **invertible** map, so it cannot change what a linear probe can express; the small movement is
  pure regularisation. Recorded because it is the textbook speaker-verification answer, and it is
  worth stating plainly that on its own it cannot work here.
* **`CORAL`** — the only second-order family and the only one that reaches the real leak (F3):
  it is the sole transform here that moves the MLP probe (0.949 → 0.694).
  Transductive `CORAL-all` pays for it with the task (0.516) — compression.
  Inductive `CORAL-train` keeps valence (0.651) but only because it leaves the
  target lab un-aligned, which is also why its lab column cannot be read as a deployment number († in §4).
  A second-order method that aligns an *unseen* site without its statistics is not something this
  family provides.
* **Per-lab standardisation (transductive)** — center 0.537, scale
  0.630, full z-score 0.506. This is what a
  practitioner could actually deploy given unlabelled audio from a new site, and on this corpus **it
  is the worst thing you can do** — the only significantly harmful method in the study (F4). Their
  lab-probe numbers are near-vacuous by construction (the normalisation is *defined* by the lab
  label); the meaningful column for them is valence. Reported separately and labelled transductive
  throughout.

## 7. Verdict

**Site invariance is not achievable post hoc on these frozen embeddings, and the obstruction is
structural rather than a matter of finding a better eraser.**

1. The first-order part is *solved* — closed form, rank 5, guarantee holds empirically. It buys
   +0.003 valence.
2. What actually carries site is **second-order** (covariance), which no erasure method targets; the
   one method that does (CORAL) removes the task with it.
3. Site and valence are confounded by corpus design — 0.766 balanced accuracy for
   valence from *the lab name alone*, against 0.654 for the embedding — so any transform strong
   enough to remove site removes signal.
4. A large share of the leave-one-lab-out penalty is unseen-**context** generalisation, which site
   invariance cannot address at all.
5. The evaluation is underpowered for the effect sizes being chased: 4 scorable folds, baseline CI
   width 0.199, and a random projection that touches nothing wanders ±0.013.

This is not a post-hoc representation-surgery problem. It is a **study-design and
encoder-training** problem: record shared contexts across sites so site and task are decorrelated,
or intervene during encoder training with a second-order-aware objective. Both are outside what
frozen-embedding erasure can reach.

## 8. Honest limits

* One corpus, one encoder, one layer (layer 0, chosen by prior work as best for pigs). Nothing here
  shows the second-order story is layer- or encoder-invariant.
* Only 4 of 6 labs are scorable for valence; the small denominator and the confound have
  the same cause. Absolute valence numbers carry a CI roughly 0.20 wide.
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
