# Conformal coverage is marginal. Animals are not.

First run of `conformal_pam/src/conformal.py` on real data. Until now that module
had only ever been validated on synthetic scores. This connects it to the audit
finding that per-animal accuracy varies enormously and that identity is highly
decodable.

**Hypothesis.** Marginal (average) conformal coverage will look fine while
per-individual coverage is catastrophically uneven.

**Result: confirmed, on all three datasets, at both layers tested.** Marginal
coverage lands on the nominal 0.90 to within 0.01 while per-individual coverage
runs from 0.41 to 1.00. Mondrian conditioned on class does not fix it. Mondrian
conditioned on group does fix it — but only for groups you already have labelled
data from, which in deployment is exactly the group you do not have.

Everything below is `alpha = 0.10`, nominal coverage `0.90`, frozen
WavLM-base-plus layer 9 (layer 6 as robustness), cached embeddings only.

---

## Setup

| | cats | dogs | pigs |
|---|---|---|---|
| clips | 348 | 308 | 5031 |
| grouping unit | individual | individual | recording lab |
| groups | 20 | 10 | 6 |
| group sizes | 4–52 | 15–47 | 354–1466 |
| leave-one-group-out probe accuracy (L9) | 0.790 | 0.857 | 0.634 |

Pipeline: logistic probe (`C=1`, `class_weight=balanced`, the project's own
`fit`), leave-one-group-out so every clip carries an **out-of-fold** probability
from a probe that never saw its own animal or lab. Embeddings z-scored within
species (cats, dogs) or globally (pigs), matching `cross_species.zscore_within`
and `pigs_run` part A.

The binary affect problem is turned into a set-valued predictor: score matrix
`S = [P(positive), P(negative)]`, one-hot `Y`, a label enters the set iff its
score clears the conformal threshold. So the prediction set is a subset of
{positive, negative}: **size 1 = a committed call, size 2 = "I don't know",
size 0 = "neither label is plausible"**. With one-hot `Y`,
`split_conformal_threshold` is exactly the LAC/THR conformal classifier and
`coverage` is exactly P(true label ∈ set).

Two calibration regimes, each pooled over 40 repeated splits (pigs: all 20
possible 3-vs-3 lab partitions) so no number rests on one lucky draw:

* **A — group-disjoint.** Calibrate on half the individuals/labs, test on the
  held-out half. This is deployment: the animal in front of you is new.
  Exchangeability is broken on purpose.
* **B — within-group.** Random 50/50 split inside every group. This is the
  exchangeable control, and the only regime in which group-conditional Mondrian
  has any calibration data to condition on.

---

## 1. The headline: marginal is fine, per-group is not

Design A, pooled split conformal, layer 9:

| | marginal coverage | per-group min | median | max | IQR | groups < 0.80 | mean set size |
|---|---|---|---|---|---|---|---|
| cats | **0.904** | **0.412** | 0.913 | 1.000 | 0.186 | **5 / 20** | 1.32 |
| dogs | **0.892** | **0.624** | 0.958 | 1.000 | 0.059 | **2 / 10** | 1.10 |
| pigs | **0.891** | **0.735** | 0.913 | 1.000 | 0.079 | **1 / 6** | 1.67 |

Per-group **set size** is uneven too — cats 1.01–1.55, dogs 1.00–1.24, pigs
1.52–1.93. Some animals get a committed answer on nearly every call; others get
"I don't know" on nearly half.

Worst offenders:

* `cat_TIG01` 0.412, `cat_NIG01` 0.611, `cat_MEG01` 0.748, `cat_BRA01` 0.760,
  `cat_IND01` 0.795
* `dog_Freid` 0.624, `dog_Luke` 0.721 — while `dog_Rudy` gets 1.000 and
  `dog_Mac` 0.992
* pigs `IASPB` 0.735 — while `IASPC` gets 1.000

A practitioner told "this detector has 90% coverage" and handed `dog_Freid`
gets 62%. The gap between the promise and the delivery is 28 points, and
nothing in the marginal number reveals it.

## 2. This is not a bug in the conformal code

Three independent checks, all clean:

* **Synthetic exchangeable control.** 2000 trials, n=400, exchangeable by
  construction: mean coverage **0.9048** (target 0.900), sd 0.029. (37% of
  individual trials land below 0.90 — expected; the guarantee is in expectation
  over calibration draws, not per draw.)
* **Real scores, exchangeable split** (design B pooled, random split inside each
  group): cats **0.907**, dogs **0.903**, pigs **0.901**.
* **The module's two thresholding routines agree.** With one-hot labels the
  per-window FNR is `1 − covered`, so `conformal_risk_control` and
  `split_conformal_threshold` bound the same quantity, and they do:
  dogs t = 0.188 vs 0.205, test FNR 0.076 vs 0.082; pigs t = 0.0074 vs 0.0070,
  FNR 0.094 vs 0.092.

The implementation delivers exactly what it promises. **What it promises is
marginal coverage, and marginal coverage is not what a field biologist needs.**
The per-group spread under design B is essentially as bad as under design A
(cats IQR 0.204 vs 0.186; dogs 0.068 vs 0.059; pigs 0.065 vs 0.079), so the
unevenness is *not* an artefact of breaking exchangeability. It is the
conditional-coverage gap that split conformal never claimed to close.

## 3. Comparing calibration schemes

Layer 9, per-group coverage spread. `own θ` = how many groups had enough of
their own calibration data to support a 90% guarantee (needs n ≥ 9; we require
10).

| dataset | scheme | regime | marginal | min | IQR | < 0.80 | set size | own θ |
|---|---|---|---|---|---|---|---|---|
| cats | (a) pooled | A | 0.904 | 0.412 | 0.186 | 5/20 | 1.32 | – |
| cats | (b) Mondrian / class | A | 0.907 | 0.425 | 0.154 | 4/20 | 1.36 | – |
| cats | (b') Mondrian / predicted class | A | 0.916 | 0.746 | 0.138 | 4/20 | 1.36 | – |
| cats | (c) Mondrian / group | A | 0.904 | 0.412 | 0.186 | 5/20 | 1.32 | **0/20** |
| cats | (c) Mondrian / group | B | 0.903 | 0.467 | 0.158 | 6/20 | 1.33 | **6/20** |
| dogs | (a) pooled | A | 0.892 | 0.624 | 0.059 | 2/10 | 1.10 | – |
| dogs | (b) Mondrian / class | A | 0.888 | 0.624 | 0.092 | 2/10 | 1.18 | – |
| dogs | (c) Mondrian / group | A | 0.892 | 0.624 | 0.059 | 2/10 | 1.10 | **0/10** |
| dogs | (c) Mondrian / group | B | **0.947** | **0.905** | **0.039** | **0/10** | 1.28 | **8/10** |
| pigs | (a) pooled | A | 0.891 | 0.735 | 0.079 | 1/6 | 1.67 | – |
| pigs | (b) Mondrian / class | A | 0.873 | 0.758 | 0.122 | 1/6 | 1.65 | – |
| pigs | (c) Mondrian / group | A | 0.891 | 0.735 | 0.079 | 1/6 | 1.67 | **0/6** |
| pigs | (c) Mondrian / group | B | 0.903 | **0.898** | **0.002** | **0/6** | 1.64 | **6/6** |

**(b) Mondrian by class does not help.** Per-group minimum is unchanged for cats
(0.412 → 0.425) and dogs (0.624 → 0.624), and for pigs the marginal coverage
actually drops below nominal (0.891 → 0.873) while the IQR *widens*
(0.079 → 0.122). The taxonomy is orthogonal to the source of the heterogeneity.
Conditioning on the *predicted* class (also a valid Mondrian taxonomy, since it
is a function of x alone) lifts the worst cat to 0.746, but it does so by
inflating sets, and it makes pigs worse (worst lab 0.611). Neither is a real fix.

**(c) Mondrian by group is vacuous in deployment.** Under a group-disjoint split
*no* test group has calibration data, so every group falls back to the pooled
threshold and scheme (c) is bit-for-bit identical to scheme (a). This is not a
coding shortcut; it is the structural fact. Group-conditional conformal cannot
condition on a group it has never seen.

**(c) Mondrian by group repairs it completely once the group has its own
labels** — where there is enough data:

* dogs: worst individual 0.696 → **0.905**, IQR 0.068 → 0.039, groups below 0.80
  goes 2 → **0**. `dog_Freid` 0.696 → 0.962; `dog_Luke` 0.730 → 0.956.
* pigs: worst lab 0.775 → **0.898**, IQR 0.065 → **0.002**. Every lab lands on
  0.90 to within half a point.
* cats: **no repair.** Worst individual 0.467 → 0.467, still 6/20 below 0.80.

The cats failure is a sample-size wall, and it is the most practically important
number here. At `alpha = 0.10` you need at least 9 calibration clips from *that
individual*; with a 50/50 split that means ≥ 20 clips in total. Only **6 of 20
cats** clear it. The other 14 fall back to pooled and keep their bad coverage.
`cat_TIG01`, the worst animal in the study, has 5 clips — you cannot make it a
90% promise no matter what method you use.

**The repair is paid for in set size, and paid unevenly.** Mondrian by group
widens the sets exactly on the animals the model is bad at and tightens them on
the animals it is good at. Dogs, design B, per-group mean set size: `Freid`
1.77 and `Luke` 1.91 (near-total abstention) against `Rudy` 0.91 and `Zoe` 0.96
(below 1 — i.e. some empty sets). Pigs: `IASPB` 1.86, `IASPC` 0.98 down from
1.71. That is the honest description of what group-conditional conformal buys —
not better predictions, but an honest accounting of where the model is useless.

## 4. Species shift breaks the marginal guarantee outright

Calibrate on one species, test on the other (single fixed probe: trained on half
the source individuals, calibrated on the other half, applied to the whole
target species; 20 repeats).

| | marginal coverage | per-group min | IQR | groups < 0.80 | set size | same-species reference |
|---|---|---|---|---|---|---|
| dogs → cats, L9 | **0.747** | 0.330 | 0.146 | 12 / 20 | 1.33 | 0.922 |
| cats → dogs, L9 | **0.746** | 0.546 | 0.251 | 5 / 10 | 1.46 | 0.920 |
| dogs → cats, L6 | **0.556** | 0.205 | 0.227 | 16 / 20 | 1.21 | 0.914 |
| cats → dogs, L6 | **0.780** | 0.686 | 0.111 | 5 / 10 | 1.59 | 0.923 |

Here even the *marginal* promise fails — a nominal 90% detector delivering 75%
(and 56% at layer 6). Note the contrast with §1: changing individuals within a
species costs essentially nothing marginally (0.89–0.90); changing species costs
15–35 points. Unweighted split conformal has no defence against this, which is
the concrete argument for `weighted_conformal_threshold` — untested here, and it
would need a defensible likelihood-ratio estimate, which is its own failure mode.

## 5. Layer 6 robustness

Same story, different numbers. Design A pooled: cats marginal 0.903 / worst
0.600 / 4 of 20 below 0.80; dogs 0.890 / 0.734 / 1 of 10; pigs 0.894 / 0.701 /
1 of 6. Design B Mondrian by group again repairs dogs (worst 0.847, IQR 0.023)
and pigs (worst 0.899, IQR 0.009) and again fails cats (worst 0.558). No
conclusion depends on the layer choice.

---

## Honest caveats

1. **Small cat groups make the extreme numbers coarse.** `cat_TIG01` has 5 clips,
   so 0.412 means roughly 2 of 5 covered. Restricted to the 15 cats with n ≥ 10,
   the worst is 0.748 and the IQR 0.113, with 3 of 15 below 0.80. The effect
   survives the restriction but is less dramatic than the headline minimum
   suggests. Dogs and pigs have no groups this small.
2. **Out-of-fold scores are not textbook split conformal.** Calibration and test
   clips carry probabilities from different leave-one-group-out probes, which is
   a cross-conformal-flavoured approximation rather than a single frozen scorer.
   The synthetic control and the design-B marginal coverages (0.901–0.907)
   indicate this costs nothing here, but it is a deviation and should be stated.
3. **z-scoring uses all clips of a species, test included.** Unsupervised
   per-feature standardisation, consistent with the rest of the project, but
   mildly transductive.
4. **Group coverage is confounded with class prior.** The worst pig lab `IASPB`
   is 100% positive-valence and `IASPC` is 100% negative — valence is confounded
   with recording team in this dataset by construction. Several cats are
   single-context too (`BLE01`, `CLE01`, `MEG01` all negative; `BRI01`, `IND01`,
   `JJX01` all positive). So "per-individual coverage failure" here is partly
   per-individual *label-prior* shift, not purely an identity-acoustics effect.
   Distinguishing the two would need groups with matched class balance.
5. **Pigs' sets are barely informative to begin with.** Mean set size 1.67 means
   the predictor abstains on two thirds of calls — unsurprising at 0.63 probe
   accuracy. The coverage guarantee is being met largely by refusing to answer.
6. `conformal_risk_control`, `weighted_conformal_threshold`, `temperature_scale`
   and `platt_scale` were not exercised beyond the agreement check in §2.
   Conformal does not need calibrated probabilities, so temperature scaling was
   skipped deliberately.

## What this is worth

The two chapters do connect, and the connection is a negative result that
motivates the positive one. Split conformal gives an honest marginal guarantee
on real bioacoustic embeddings — it is not broken. But per-animal coverage
ranges 0.41–1.00 while the average sits at 0.90, so the guarantee is silent
about the animal actually in front of the microphone. Mondrian conditioned on
class does not close that gap. Mondrian conditioned on group closes it perfectly
— and requires ~20 labelled calls from the specific individual, which is
precisely what you do not have for a new animal and cannot obtain for a rare
one. The deployable claim is therefore narrower than the chapter would like:
*group-conditional conformal turns an unearned average promise into an honest
per-animal one, at the cost of abstaining on the animals the encoder cannot
read, and it is only available where you can afford to label the animal first.*

---

## Files

* `oof.py` — stage 1, leave-one-group-out out-of-fold probabilities from cached
  embeddings (no audio, no encoders, no torch). Writes `oof_L{9,6}.npz`,
  `shift_L{9,6}.npz`.
* `analyse.py` — stage 2, conformal designs A/B, shift, sanity. Writes
  `results.json`.
* `plot.py` — stage 3. Writes `coverage.png`.
* `results.json` — every number above, plus full per-group tables.
* `coverage.png` — per-group coverage strip plots, schemes side by side, 0.90
  target line, species-shift row.

Reproduce: `python oof.py && python analyse.py && python plot.py` (about 6 s
total on the M2; BLAS threads pinned to 2).
