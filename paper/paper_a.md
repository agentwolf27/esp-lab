# Who, Not Why: Individual and Site Identity Dominate Frozen-Encoder Evaluation of Animal Vocalisations

*Draft v0.1 — 17 Aug 2026. Target: ICBINB-BIO @ NeurIPS 2026 (deadline 29 Aug, 11:59 AoE).*
*Every number in this draft traces to a JSON file in `results/`. Placeholders are marked ⟨⟩.*

---

## Abstract

Frozen audio encoders have become the default feature extractor for animal-vocalisation tasks, and
they are routinely evaluated with random train/test splits. We show this measures the wrong thing.
Across four public corpora — cat meows (21 individuals), dog barks (10), pig calls (6 recording labs)
and wild Egyptian fruit bats (10 emitters) — and six feature sets, individual and site identity are far
more decodable than behavioural context (identity 0.63–0.94 against chance 0.05–0.17), and random
splits inflate reported context accuracy by 10 to 24 points relative to held-out-group splits. Within
a corpus, the feature set that leaks less identity inflates less (partial *r*=0.646, *p*=7×10⁻¹¹),
though this does not hold as a cross-corpus law (*r*=0.221, *n*=12, n.s.). Two consequences are
concrete: on pig valence under held-out-lab evaluation, 88 classic acoustic descriptors (eGeMAPS, 0.669)
*outperform* a self-supervised speech encoder (WavLM, 0.660) while leaking less identity; and
the originating study's own feature set scores below chance (0.386) once a lab is held out. The failure
extends to uncertainty: conformal prediction at α=0.10 returns on-target marginal coverage (0.89–0.90)
while the worst individual receives 0.412, and the standard group-conditional remedy is structurally
unavailable under a group-disjoint split. We quantify how much target-group labelling repairs each
failure and recommend reporting identity decodability alongside every context result.

---

## 1. Introduction

Two conventions have become standard in computational bioacoustics. First, use a large pretrained
audio encoder — BirdNET, Perch, AVES, BEATs, or a speech model such as WavLM — frozen, as a feature
extractor. Second, evaluate the resulting classifier on a random split of the available clips.

The first convention is well justified: these models are strong, cheap to run, and available. This
paper is about the second. Animal-vocalisation corpora are small and are collected from a handful of
individuals, often in a handful of sessions or labs. A random split places clips from the same animal,
recorded minutes apart in the same room, on both sides of the train/test boundary. If the encoder
represents *who is calling* and *where the recording was made* — and we show it represents both very
strongly — then a random-split evaluation rewards recognising the animal, and reports the result as
understanding of the animal's behaviour.

We audit this directly. Our protocol replaces the random split with a held-out-group split (leave one
individual out, or leave one recording lab out) and reports three numbers side by side for every
dataset and feature set: honest accuracy, random-split accuracy, and **group-identity decodability**
measured on the same features. The third number is the diagnostic; it explains the gap between the
first two.

**Contributions.**
1. A four-corpus, six-feature-set audit under one honest protocol (§4).
2. The relationship between identity leakage and evaluation inflation, scoped precisely: it holds
   within a corpus and is *not* established across corpora (§4.2).
3. An unexpectedly strong simple baseline — eGeMAPS matches or beats a self-supervised encoder once
   evaluation is honest (§4.3) — and a published feature set that falls below chance under site shift.
4. A control showing that a widely used dataset's context label is partly predictable from the
   *background* alone, because the label is confounded with the recording room (§4.4).
5. A channel stress test: the context axis is 1.4–4.2× more robust than identity to degradation (§5).
6. The failure propagates to uncertainty quantification: per-group conformal coverage collapses while
   marginal coverage looks correct, and the textbook remedy is vacuous in deployment (§6).
7. Recovery curves giving the labelling budget at which each failure is repaired (§7).

---

## 2. Related work

⟨To be completed from `RELATED_WORK.md` — verification agent running. Must include: the mature
speaker-independence literature in speech emotion recognition; shortcut learning / dataset bias;
BEANS and BirdSet evaluation conventions; prior critiques of the specific corpora; and conformal
prediction in audio.⟩

---

## 3. Protocol

**The three numbers.** For every (dataset, feature set, layer) we report:
- **honest** — balanced accuracy on the context/valence task with an entire group held out;
- **leaky** — balanced accuracy under a random stratified 5-fold split of the same clips;
- **identity** — accuracy of predicting the group label (which individual, which lab) from the same
  features under a random split.

**Groups.** Individual animal for cats, dogs and bats; recording lab for pigs. The group is the unit
that must not straddle the split.

**Probes.** ℓ2-regularised logistic regression on standardised, mean-pooled embeddings, one layer at a
time. We use a linear probe deliberately: it measures what is linearly available in the representation
rather than what a trained head can extract, and it is the setting practitioners actually use.

**Feature sets.** WavLM-base-plus, HuBERT-base, wav2vec2-base, AVES-bio (bioacoustic SSL), eGeMAPSv02
(88 openSMILE functionals), and an 85-dimensional MFCC/log-mel summary. All frozen.

---

## 4. The audit

### 4.1 Main table

⟨Insert `figures/audit_table.png` and the table from `results/out_audit/results.json`.⟩

Inflation ranges from +0.099 to +0.242. It is present in every dataset and every feature set,
including the two hand-crafted ones — this is not a neural-network pathology, it is an evaluation
pathology.

### 4.2 Identity leakage predicts inflation — within a corpus

Pooling all (dataset, feature set, layer) points, identity decodability correlates with the inflation
gap at *r*=0.611 (*n*=83, *p*=9×10⁻¹⁰); the within-dataset partial correlation is *r*=0.646
(*p*=7×10⁻¹¹). It survives restricting to layer-only variation inside a single encoder (*r*=0.636),
holds in all six encoder clusters (sign test *p*=0.031), and is not a dimensionality artifact
(neural-only *r*=0.694). Identity tracks the *leaky* number (*r*=0.754) more than the honest one
(*r*=0.509), as the leakage account requires.

**The limit.** Across the 12 genuinely independent dataset×feature-set units the correlation is
*r*=0.221 (*p*=0.489). We therefore claim only: *among feature sets and layers for a given dataset,
the one that leaks less identity inflates less.* We do not claim that corpora with more identity
leakage inflate more.

### 4.3 An unexpectedly strong simple baseline

On pig valence under held-out-lab evaluation, eGeMAPS scores **0.669** and WavLM **0.660** (best layer; mean over layers 0.604), while
eGeMAPS leaks less identity (0.865 vs 0.938). Under a random split WavLM appears clearly ahead
(0.883 vs 0.823). The encoder's advantage on this dataset exists only in the leaky number.

Separately, the 18 acoustic features published with the pig corpus score **0.386** under held-out-lab
evaluation — below the 0.500 chance level — indicating that the feature-to-valence mapping partly
inverts between labs. ⟨Note: our re-analysis uses a linear probe; the original study used other
classifiers and did not claim cross-lab generalisation. This is a reframing of their features under a
different protocol, not a refutation of their result.⟩

### 4.4 The label can be predicted from the background

CatMeows induces its isolation condition by moving the cat to an unfamiliar room. Discarding the
vocalisation and keeping only the quietest 30% of frames still yields 0.725 on the binary task —
approximately the full-embedding number. A reverberation proxy separates the classes at +0.10/−0.18 SD
(cats) and +0.67/−0.32 SD (dogs). Part of what is called context classification on this corpus is room
classification.

---

## 5. Channel robustness

Perturbing the same clips (4 kHz low-pass, 0.4 s reverb, +10 dB SNR noise) and measuring decision
flips against probes trained on clean audio: at layer 9, context flips 1.1% under bandwidth loss
against 4.7% for a **difficulty-matched** identity probe (4.2×); reverb 1.9×; noise 1.4×. Matching
difficulty matters — the unmatched comparison exaggerates the ratio to 2.7–6.1×. Gain is invariant by
construction (waveform normalisation) and is reported as non-informative.

Interpretation: individual identity lives in fine spectral detail, affect in coarse envelope
structure. This also constrains §4.4 — if the context signal were purely room acoustics, reverb would
scramble it, and it does not.

---

## 6. The guarantees break too

Split conformal prediction at α=0.10 on out-of-fold scores, calibrated and tested by group:

| dataset | marginal | worst group | IQR | groups < 0.80 |
|---|---|---|---|---|
| cats (20) | 0.904 | **0.412** | 0.186 | 5/20 |
| dogs (10) | 0.892 | 0.624 | 0.059 | 2/10 |
| pigs (6 labs) | 0.891 | 0.735 | 0.079 | 1/6 |

This is the conditional-coverage gap, not an implementation error: a synthetic exchangeable control
returns 0.9048 over 2,000 trials, and the per-group spread is equally wide under an exchangeable split.

**The standard remedy is unavailable.** Mondrian conformal conditioned on the group is bit-for-bit
identical to pooled under a group-disjoint split, because no test group has calibration data (verified:
0/20, 0/10, 0/6 groups receive their own threshold). Conditioning on predicted class does not help
(worst cat 0.412→0.425). Given some of the deployment group's own labels, repair is complete where
data allows (dogs 0.696→0.905; pigs 0.775→0.898, IQR→0.002) but requires **≥9 labelled clips at
α=0.10**, which only 6 of 20 cats possess.

**Shift.** Calibrating on dogs and testing on cats collapses even marginal coverage to 0.747 (L9) and
0.556 (L6). Swapping individuals within a species costs nothing marginally; swapping species costs
15–35 points.

---

## 7. How much labelling repairs it

Adding *k* labelled clips from the held-out group back into training:

| | k=0 | k=10 | k=25 | k=50 | k=100 | random ceiling |
|---|---|---|---|---|---|---|
| cats, WavLM | 0.576 | 0.677 *(eff. 7.7)* | **0.727** *(eff. 10.9)* | 0.726 | 0.727 | 0.729 |
| cats, eGeMAPS | 0.525 | 0.581 | 0.587 | 0.592 | 0.592 | 0.628 |
| pigs, WavLM | 0.654 | 0.679 | 0.707 | 0.730 | 0.769 | 0.884 |
| pigs, eGeMAPS | 0.677 | 0.682 | 0.686 | 0.693 | 0.704 | 0.823 |

Two different shapes. **Cats saturate at ≈11 effective clips** and reach 0.727 against a 0.729
ceiling — the individual gap closes completely, and it cannot be pushed further because the median cat
contributes only ~19 clips in total. **Pigs do not saturate**: 100 clips from the target lab close
50.2% of the gap and the curve is still climbing, so lab shift is a deeper problem than individual
idiosyncrasy.

The two feature sets behave differently again. On pigs, eGeMAPS *starts ahead* (0.677 vs 0.654 at
k=0) but gains only +0.027 from 100 clips against WavLM's +0.115. The encoder is more **adaptable**,
not more **transferable**. The conformal analysis in §6 independently arrives at ≈9 clips as the
threshold for a per-group guarantee, from an entirely different direction.

---

## 8. Limitations

- One encoder family (speech SSL) dominates the comparison; AVES is the only bioacoustic encoder tested
  at scale, and the newer AVEX checkpoints are untested.
- Bats have **no decodable context label** in the available packaging (0.301 vs 0.250 chance) and are
  excluded from the inflation fit; they contribute only the identity result.
- Dog context is confounded with recording session; our dog numbers are leave-one-individual-out, not
  leave-one-session-out.
- The cross-corpus inflation correlation is null (*n*=12); we do not claim a general law.
- Probes are linear and layers are selected post hoc for the "best layer" column; the mean-over-layers
  column is selection-free and we report both.
- Two runs of the cats/eGeMAPS number differ slightly (0.486 in a direct run, 0.492 in the audit
  pipeline) because of regularisation and fold-seed differences; we quote the audit value throughout
  and the discrepancy is well inside the ~1–3 point noise floor established in `results/out_multiband/`.
- Extreme per-group conformal minima come partly from small groups; restricted to cats with n≥10 the
  worst is 0.748 and 3/15 fall below 0.80 — the effect survives, less dramatically.

---

## 9. Recommendations

1. **Split by group, never randomly.** The group is the individual or the recording site.
2. **Report identity decodability next to every context number.** It is one extra probe and it tells
   the reader how much room the evaluation left for leakage.
3. **Report per-group spread, not just the mean** — for accuracy and for coverage.
4. **Include a hand-crafted baseline** (eGeMAPS). Under honest evaluation it is sometimes ahead.
5. **State the target-group labelling budget** if you claim a per-group guarantee: below ≈9 clips at
   α=0.10, no such guarantee is available.

---

## Reproducibility

All code, cached results and figures: `github.com/agentwolf27/esp-lab`. Every experiment runs on a
CPU laptop; total compute cost for the paper is zero GPU-hours. Datasets are public: CatMeows
(Zenodo 4008297, CC-BY-4.0), dog barks (Molnár et al. 2008, via BEANS), Soundwel (Zenodo 8252482,
CC-BY-4.0), Egyptian fruit bats (Prat et al. 2017, via BEANS).
