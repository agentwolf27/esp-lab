# Overnight run — findings, 17 Aug 2026

Everything below was run on the laptop (Apple M2, CPU only) between roughly 04:30 and 07:00.
No GPU, no cluster. Raw outputs, code and figures:
`experiments/context_probe/` (code), `results/` (JSON), `figures/` (plots).

---

## 0. One-paragraph summary

Frozen **human-speech** encoders classify a cat's behavioural context better than hand-crafted
features (0.571 vs 0.399, chance 0.333) and better than ESP's own bioacoustic encoder AVES (0.540).
But most of that within-species signal is **not the animal** — the quiet background frames alone
reach 0.725 on the binary task, because CatMeows induced isolation by moving the cat to a different
room. The result that *does* survive every control is cross-species: an affect direction learned on
**dogs** (aggression vs play) predicts **cat** contexts (isolation vs brushing) at 0.583 mean over
all layers, p=0.0033, replicated in a second encoder, rising monotonically with depth, and **not**
reproducible from the background or from simple acoustics. The reverse direction (cat→dog) does not
survive. Most striking: duration is the only simple feature that points the same way in both species
— energy, spectral centroid and zero-crossing rate all point in **opposite** directions — yet the
deep embedding still finds a transferable direction. So whatever transfers is *not* "loud, long and
high-pitched means distress."

---

## 1. Experiment 1 — within-species context (CatMeows)

440 meows · 21 cats · 3 contexts (brushing / waiting-for-food / isolation) · leave-one-cat-out ·
chance 0.333.

| encoder | best layer | context (held-out cat) | context (random split) | cat identity (21-way) |
|---|---|---|---|---|
| hubert-base-ls960 | 12 | **0.571** | 0.671 | 0.700 |
| wavlm-base-plus | 3 | 0.559 | 0.725 | 0.793 |
| wav2vec2-base | 2 | 0.554 | 0.737 | 0.755 |
| **AVES-bio (ESP)** | 5 | 0.540 | 0.736 | 0.793 |
| MFCC baseline (85-d) | – | 0.399 | 0.602 | 0.768 |

**(a) Speech encoders beat hand-crafted features.** +0.17 over MFCC. Solid.

**(b) Speech encoders beat ESP's bioacoustic AVES** (0.571 vs 0.540). Consistent with *Crossing the
Species Divide* (DCASE 2025), which found frozen speech models rival fine-tuned bioacoustic models on
BEANS. Here it holds on a task none of them were built for. Caveat: one dataset, one species, and
AVES-bio is the 2023 model — the newer AVEX `esp_aves2_*` checkpoints were not tested.

**(c) Random splits inflate massively.** Same features, same classifier: 0.399 → 0.602 (MFCC),
0.559 → 0.725 (WavLM). The reason is in the last column — these features identify *which of 21 cats*
is meowing at 0.70–0.79 accuracy (chance 0.048). Any split that lets a cat appear in both train and
test is measuring voice recognition.

**(d) Identity fades with depth; context does not.** WavLM identity falls 0.820 (L1) → 0.655 (L12)
while context stays ~0.51–0.56. The context/identity ratio is best at the deepest layers. If you want
an encoder that hears the *situation* rather than the *animal*, go deep.

**Confusion matrix (MFCC, held-out cat):** brushing 0.44, isolation 0.44, **waiting-for-food 0.29 —
below chance**, mostly misfiled as isolation. The only axis with real signal is calm-at-home versus
alone-in-a-strange-room. That is an **arousal** axis, not a lexicon.

---

## 2. THE BIG CAVEAT — the room, not the cat

CatMeows induces isolation by **moving the cat to an unfamiliar room**. So the negative class differs
by recording environment *by construction*. Test: discard the vocalisation, keep only the quietest
30% of frames, rerun.

| | background-only | full embedding |
|---|---|---|
| within-species, cat (binary) | **0.725** | 0.69–0.79 |
| within-species, dog (binary) | 0.614 | 0.74–0.89 |
| transfer cat→dog | 0.511 (p=0.460) | 0.527 |
| transfer dog→cat | 0.553 (p=0.113) | **0.583 (p=0.0033)** |

**The background alone does about as well as the full embedding within cats.** A large share of
"context classification" on this dataset is room/channel, not voice. Reverb proxy confirms it: the
energy-decay slope separates classes at +0.67/−0.32 SD in dogs and +0.10/−0.18 in cats.

This is a caveat for Experiment 1 and, separately, **a finding in its own right** — it is the
leakage-audit idea landing on real data. Published accuracies on CatMeows are much higher than ours;
before believing any of them, check both the split *and* whether the room is doing the work.

Crucially, **the background does not transfer across species** (p=0.113), so §3 survives this.

---

## 3. Experiment 3 — cross-species affect transfer

Added dog barks (Molnár 2008, via ESP's BEANS `dogs`): 693 barks, 10 named dogs.
Binary affect axis, chosen to be defensible in both species:

- **negative / agonistic-distress** = cat isolation (221) + dog aggression (99)
- **positive / affiliative** = cat brushing (127) + dog play (209)
- excluded as ambiguous: cat waiting-for-food, dog contact

656 clips, 20 cats + 10 dogs. Z-scored **within species** (removes the species offset — without this
you are partly just detecting cat-vs-dog). Train on all of species A, test on all of species B.
300-permutation null. Chance 0.500.

**Selection-free statistic — mean over all 13 layers, nothing chosen after seeing the answer:**

| encoder | dog→cat | null | p | cat→dog | p |
|---|---|---|---|---|---|
| wavlm | **0.583** | 0.490±0.043 | **0.0033** | 0.527 | 0.283 |
| hubert | **0.562** | 0.490 | **0.0367** | 0.522 | 0.293 |
| wav2vec2 | 0.530 | 0.495 | 0.210 | 0.551 | 0.090 |

Per-layer dog→cat (WavLM) is significant at L7 (0.624, p=0.013), L9 (0.656, p=0.000),
L10 (0.646, p=0.007), L11 (0.632, p=0.030), L12 (0.623, p=0.017) — **a monotonic rise with depth, not
an isolated spike.** cat→dog reaches p<0.05 only at L3 and is noise elsewhere.

Bootstrap over target individuals (resampling whole animals): hubert best layer 0.631, 95% CI
[0.582, 0.676] — excludes 0.5. Per-cat consistency 12/19 above chance (median 0.579).

**Verdict: dog→cat is probably real. cat→dog is not established.**

---

## 4. The most interesting result — simple acoustics point the *opposite* way

Control: 4 interpretable features (log duration, log energy, spectral centroid, zero-crossing rate).

Within species they work (cat 0.638, dog 0.643). **Across species they fail completely**:
cat→dog 0.459 (p=0.607), dog→cat 0.448 (p=0.683).

Per-feature separation, negative minus positive, in SD:

| feature | cat (isolation − brushing) | dog (aggression − play) | same direction? |
|---|---|---|---|
| log duration | +0.28 | +0.31 | **yes** |
| log energy | +0.69 | −0.19 | no |
| spectral centroid | +0.12 | −0.47 | no |
| zero-crossing rate | +0.14 | −0.48 | no |

Only duration agrees. The textbook arousal cues (louder, higher, harsher) **invert between these two
species on these two contrasts** — dog play is louder and harsher than dog aggression, while cat
isolation is louder and harsher than cat brushing. And yet the deep embedding still transfers.

So the transferable structure is **not** the classical acoustic-universals story. That is the single
most publishable sentence of the night, and also the one most in need of a third species.

---

## 5. What would still kill it

Not yet controlled, in rough order of danger:

1. **Different corpora, different labs, different microphones.** Cats (Italy, 2021) and dogs
   (Hungary, 2008) were recorded by different teams. Within-species z-scoring removes the mean
   offset but not a class-correlated channel difference that happens to align across corpora.
   *Settles it:* a third species from a fourth lab; or matched-channel augmentation (convolve both
   corpora with the same set of impulse responses and re-run).
2. **Only two species, one contrast each.** The "affect axis" is really "isolation-vs-brushing"
   mapped onto "aggression-vs-play". Those may share something specific rather than general.
   *Settles it:* bats, pigs, zebra finches — datasets that carry several contexts each.
3. **Ten dogs, twenty cats.** Individual-level bootstrap CIs are wide.
   *Settles it:* more individuals, or a mixed-effects model with animal as a random effect.
4. **Class imbalance interacting with `class_weight='balanced'`.** Cat is 221/127 negative-heavy,
   dog is 99/209 negative-light — the imbalance is *inverted* between species, which is exactly the
   configuration where a badly-calibrated probe can look like it transfers.
   *Settles it:* subsample to equal class sizes and rerun; report both.
5. **Transductive z-scoring.** The target-species mean/SD uses test data. Standard in domain
   adaptation, but worth reporting a strict version that z-scores the target using held-out-animal
   statistics only.
6. **Excluded categories.** Dropping cat-food and dog-contact was a judgement call made before
   seeing results, but it is still a researcher degree of freedom. *Settles it:* report the 3-class
   and 4-class variants too.

---

## 6. Housekeeping / state

- Environment: python 3.12, torch 2.13, transformers 5.15, CPU only.
- Datasets: CatMeows (8.9 MB), dog barks (901 MB); later pigs (326 MB) and bats (5.2 GB, subsampled). See `data/README.md`.
- Total compute cost: **$0**.

## 7. Files

```
ctx/
  context_probe.py      experiment 1, MFCC baseline           (stock anaconda python)
  encoders.py           experiment 1b, 5 frozen encoders      (venv)
  cross_species.py      experiment 3, cat<->dog transfer
  validate_transfer.py  permutation tests, nested selection, controls
  replicate.py          3-encoder replication + bootstrap
  room_control.py       the background/room confound test
  figures.py            all six figures
  out_enc/ out_cross/ out_fig/   results json, cached embeddings, PNGs
```

---

# ADDENDUM — adversarial review (Fable 5 synthesis) and what it changed

A senior-reviewer pass over the numbers above found three things I had wrong or overstated.
Recording them here because they matter more than the results did.

## R1. The headline was the wrong result

I led with cross-species transfer. The reviewer's judgement — and I now agree — is that the
**identity-leakage finding is the solid one** and the transfer is "suggestive, not established."
Leakage is replicated across all five feature sets, is mechanistically coherent, needs no new data,
and is already chapter 4 of the plan ("random splits leak recorder/site/time; nobody measured").
We have now measured a fourth leak axis: **individual**.

## R2. "Embeddings transfer where acoustics cannot" was overstated

**log-duration is SAME-SIGN in both species** (cat +0.28 SD, dog +0.31 SD). A same-sign feature of
d≈0.3 predicts transfer around Φ(d/2) ≈ 0.556 — which sits exactly inside the observed band
(0.530–0.583). Mean-pooled SSL states encode clip duration well. So **"the probe found duration"**
is a live and sufficient null, and I had not tested it.

The 4-feature control failed to transfer not because "handcrafted features can't", but because,
trained on cats, it loads on **energy** (d=0.69 in cats) which **anti-transfers** (−0.19 in dogs) and
drags the probe below chance. That is a *weighting* artifact, not evidence about handcrafted features
in general. The correct claim is narrower: "embeddings transfer where *this particular 4-feature
probe* does not."

Also worth noting: the sign flips are *predicted* by Morton's motivation-structural rules — the two
"negatives" are different emotions (dog hostility lowers pitch and adds roughness; cat distress
raises pitch). That makes any genuine transfer more interesting, not less.

## R3. The statistics were anti-conservative

- **Permutation unit was wrong.** 300 shuffles of *clip* labels treat 348 correlated meows as
  independent, when identity is 79% decodable. The exchangeable unit is the **animal**. At animal
  level, 12/19 cats above 0.5 is a sign test at p≈0.18 — *not* significant.
- **p-floor.** With 300 permutations the smallest reportable p is 1/301 = 0.0033. WavLM hit the
  floor; "p=0.0033" should have read "p < 0.0033". Rerunning with 2,000.
- **The asymmetry may not exist.** dog→cat 0.583 vs cat→dog 0.527, with null SDs ≈0.04, gives
  z≈0.95, p≈0.34. All three cat→dog point estimates are *above* 0.5. The honest sentence is
  "cat→dog is underpowered", never "cat→dog fails". Explaining an asymmetry before establishing it
  is the classic Gelman error.
- **b was overstated too.** HuBERT 0.571 vs AVES 0.540 is ~14 clips in a 21-cluster design where
  per-cat accuracy spans 0.00–0.86. The defensible version is the *negative*: "bioacoustic
  pretraining confers no measurable advantage over speech SSL on domestic-animal vocalisations."

## R4. And one correction to the reviewer

It proposed that identity accuracy *predicts* each encoder's inflation gap — an elegant internal
coherence. Computed: Pearson r = 0.813 but **p = 0.094**; Spearman r = 0.410, p = 0.493. With n=5
encoders this is **suggestive, not significant**. Worth a figure, not a claim.

## The two papers

**Paper A — exists now.** "Individual identity, not context: an evaluation-leakage audit of frozen
encoders for animal vocalisations." Needs: eGeMAPS baseline (the honest paralinguistics baseline,
not MFCC-85), one more individually-labelled dataset (Prat bats, 15k calls), animal-level statistics
throughout. Venue: ICBINB @ NeurIPS (negative/surprising results) — **check the deadline first**.

**Paper B — contingent on the kill-test.** Cross-species transfer. Write nothing until duration is
partialled out and animal-level permutation is run. If it dies, the autopsy ("apparent cross-species
affect transfer reduces to call duration") is itself a good ICBINB submission. Win either way.

## Ranked next experiments

1. **E1 kill-test battery** — duration-only transfer, duration-partialled embeddings, animal-level
   permutation (2,000 draws), single-feature ablations, asymmetry test, individual×class tables.
   *Running now.* Decides whether Paper B exists.
2. **E2 third species: pigs** (Briefer 2022, 7,414 calls, explicitly valence-labelled with arousal
   covariates). Turns a pair into a 3×3 matrix and separates the arousal vs valence hypothesis.
3. **E3 scale replication: Egyptian fruit bats** (Prat 2016, ~15k calls, emitter ID). Does
   identity-dominance hold on a wild, non-human-directed species at 30× the sample size?
4. **E4 placebo axis.** Train a probe on an arbitrary non-affect binary, test on cat affect. Must be
   ≈0.5, or generic corpus alignment is driving everything.
5. **E5 channel stress test.** Re-embed under low-pass / reverb / gain; do context decisions survive
   where identity decisions do not? Doubles as the thesis' trustworthiness figure.

## Two bookkeeping items to fix before anything ships

- Cat count is inconsistent across sections: 21 (Exp 1), 20 (Exp 3 header), 19 (per-cat consistency).
  Resolve which animals were dropped at each stage and why.
- State openly that within-species z-scoring is **transductive** (uses test-corpus statistics). It is
  defensible for deployment and the permutation null shares it, but unstated it reads as leakage.
  Report AUC alongside balanced accuracy — AUC is immune to intercept transfer.

---

# KILL-TEST VERDICT (E1 battery, complete)

| test | dog→cat | note |
|---|---|---|
| **duration alone** (1 feature) | **0.557** | almost exactly the predicted Φ(0.3/2)≈0.556 |
| embedding, raw (WavLM, mean over 13 layers) | 0.583 | |
| embedding, **log-duration regressed out** | **0.552** | drop of 0.031 |
| animal-level permutation, 2,000 draws (raw) | **p = 0.0045** | correct exchangeable unit; survives |
| asymmetry dog→cat − cat→dog = +0.056 | **p = 0.44** | **not significant** — never say "cat→dog fails" |
| single features: energy / centroid / zcr | 0.354 / 0.504 / 0.507 | energy anti-transfers, as predicted |

**Reading.** The transfer is real at the animal level, but roughly **half of it is duration** — the one
cue that points the same way in both species. What remains after removing duration (0.552 vs null
0.490±0.036, z≈1.7) is small and borderline; it needs its own permutation p-value and, more than
that, a third species. Combined with the earlier controls (equal class sizes 0.618 p<0.0001; strict
scaling 0.591 p=0.005), the honest one-liner is:

> *An affect axis transfers dog→cat in frozen speech embeddings and survives animal-level, balanced
> and strict-scaling tests; about half the effect is call duration, the residual is small, and the
> apparent directional asymmetry is not statistically supported.*

That is a clean, modest, ICBINB-shaped result — not the headline. The headline remains identity
leakage + the room confound (Paper A).

**Next:** third species (pigs, Briefer 2022, valence-labelled) — the only thing that can break the
two-corpus channel confound and tell arousal from valence.

---

# ITERATION 2 (afternoon 17 Aug) — specificity, residual, third species

## Placebo axes: the transfer is SPECIFIC to affect

If *any* dog binary predicted *any* cat binary, "transfer" would just be generic corpus alignment.
Mean over 13 WavLM layers, chance 0.500:

| pair | acc | verdict |
|---|---|---|
| dog AFFECT → cat AFFECT (the claim) | **0.583** | |
| dog SEX → cat AFFECT (placebo) | 0.513 | chance |
| dog AFFECT → cat SEX (placebo) | 0.498 | chance |
| cat SEX → dog AFFECT (placebo) | 0.496 | chance |
| dog SEX → cat SEX (matched, non-affect) | 0.504 | chance |

Every placebo sits at chance. Whatever transfers, it is the affect axis and nothing else. This is
the single most convincing control so far, and it costs nothing.

## The residual after removing duration IS significant (barely)

Regress log-duration out of every embedding dimension, then animal-level permutation (500 draws):
**0.552, null 0.489 ± 0.037, p = 0.048.** Intercept-free AUC: raw mean 0.673, residual mean 0.627,
deep layers 0.70–0.75. So there is real signal beyond duration — small, and it lives in the deep
layers, same place the identity signal fades.

Updated one-liner:
> *An affect axis learned on dog barks transfers to cat meows (0.583, animal-level p = 0.0045),
> survives balanced classes, strict scaling, and every placebo axis; roughly half is call duration,
> and the residual is small but significant (0.552, p = 0.048; AUC 0.63).*

## Bats: BEANS packaging has NO context labels

The BEANS `egyptian_fruit_bats` zip (5.2 GB) is 10 emitters × 1,000 calls with **Emitter only** —
the file IDs were renumbered, so the original Prat 2017 `Annotations.csv` (91,080 rows, contexts,
addressees; figshare article 4555903, downloaded) **cannot be joined** (emitter agreement 0.1%).
The full annotated audio is on figshare in ~3.6 GB chunks (article "files 207–224"). Parked.
Instead: BEANS bats is a perfectly balanced **identity** set → running the identity-decodability
replication on a wild species (2,000 calls, native-16k vs 4× time-stretch to fold 0–32 kHz into the
encoder's band — ESP's bandwidth problem #6, tested for free).

## Pigs: Soundwel is public — the real third species

Briefer et al. 2022 → **Soundwel database, Zenodo 8252482, CC-BY-4.0, 326 MB, 6,888 calls, 17
contexts with positive/negative valence.** Downloading. This is the dataset that turns one species
pair into a matrix and can separate the arousal-vs-valence hypotheses.

---

# ITERATION 3 (17 Aug, afternoon) — third species, wild species, and a rejected hypothesis

## Pigs (Soundwel, 5,031 calls after capping 700/context, 6 labs, valence Neg/Pos)

The honest unit is the **lab** (leave-one-team-out): valence is confounded with recording team.

| | held-out lab | random split | inflation |
|---|---|---|---|
| WavLM, mean over layers | **0.604** | 0.813 | **+0.21** |
| WavLM, best layer (L0) | 0.660 | | |
| **paper's own 18 acoustic features** | **0.386** (below chance) | 0.575 | |
| duration alone | 0.462 | | |

Which-lab decodability from the embedding: **0.824** (chance 0.167). Per-lab held-out valence:
IASPA 0.75, ETHZ 0.68, FBN 0.53, NMBU 0.51 — two of four labs at chance. And the published feature set
scores *below* chance when a lab is held out, meaning the feature→valence mapping partly *inverts*
between labs. That is the leakage story on the very features the paper used. (Caveat: my LOTO uses
the features as-is with a linear probe; the paper used other classifiers and did not claim cross-lab
generalisation. This is a reframing, not a refutation.)

## Bats (BEANS Egyptian fruit bats, 10 emitters × 200 calls, wild, 250 kHz)

10-way identity, chance 0.10:

| | native 16 kHz | 4× time-stretch (0–32 kHz → 0–8 kHz) |
|---|---|---|
| WavLM | 0.631 (mean 0.561) | **0.700** (mean 0.615) |
| AVES-bio | 0.651 (mean 0.629) | 0.681 (mean 0.670) |

Identity dominance is **not a domestic-animal quirk**. And a free 4× time-stretch, which folds
ultrasonic content into the encoder's band, improves identity by +7 pts (WavLM) — a small positive
result on ESP's own bandwidth problem (#6). No context labels in the BEANS packaging (see iteration 2).

## The 3×3 transfer matrix (WavLM, mean over layers, group-level permutation)

| trained → tested | cat | dog | pig |
|---|---|---|---|
| **cat** | 0.744 within | 0.527 (p=.59) | 0.524 (p=.10) |
| **dog** | **0.583 (p=.015)** | 0.811 within | 0.495 (p=.56) |
| **pig** | **0.591 (p=.005)** | 0.523 (p=.25) | 0.60 within* |

\* pig within-species shown after within-lab z-scoring is 0.53; that strips valence signal from the two
single-valence labs, so the honest number is the plain held-out-lab 0.604 above.

**Two of six off-diagonal cells are significant, and both point at cats.** Nothing transfers *into*
dogs or pigs. Placebo pig SEX → cat AFFECT: 0.510 (chance). Cats — isolation-in-a-strange-room vs
brushed-at-home — are the most predictable target for an affect axis learned on another species.

## Hypothesis tested and REJECTED: "it's isolation calls"

Pig negatives are dominated by piglet isolation; cat negatives *are* isolation. If the shared axis were
separation-distress calls, removing pig isolation from the source should collapse pig→cat.

| pig source | pig→cat | p |
|---|---|---|
| all contexts | 0.591 | .020 |
| **without isolation** | **0.568** | **.030** |
| isolation vs positive only | 0.552 | .020 |

It barely moves. Not isolation-specific. Whatever transfers is more general than that.

## Construct validity — the excluded category lands in the middle

Score every cat context with the **pig-trained** probe (fraction called negative):
brushing 0.443 → **waiting-for-food 0.495** → isolation 0.621. The category we excluded *before*
seeing results, as ambiguous, is rated as exactly ambiguous. That is the check the reviewer asked for
(E4), and it passes.

## Where this leaves the two papers

**Paper A (leakage) is now three species strong:** cats (individual, +0.10–0.20), pigs (lab, +0.21,
published features below chance across labs), bats (identity 0.63–0.70 in a wild species). Add the
bandwidth aside. This is a real, multi-dataset methods paper.

**Paper B (transfer):** two significant off-diagonal cells (dog→cat, pig→cat), placebo-clean,
survives duration partialling (borderline), balanced classes, strict scaling, group-level permutation;
excluded category intermediate; isolation hypothesis rejected. Still: only *into* cats, and only with
one encoder family. Honest framing: *a cat affect axis is predictable from probes trained on two other
species; the reverse is not; the shared component is small, partly duration, and not isolation-specific.*

## eGeMAPS — the honest handcrafted baseline (added)

openSMILE eGeMAPSv02, 88 functionals (F0, jitter, shimmer, HNR, formants…), CatMeows, leave-one-cat-out:
**0.486** (MFCC-85 was 0.399; best encoder 0.571). Random split 0.641; cat identity 0.727.
→ Claim (a) shrinks from "+0.17 over handcrafted" to **"+0.07–0.09 over eGeMAPS"** and holds. Same
leakage pattern (inflation +0.155). This is the number to quote.

---

# ITERATION 4 (17 Aug, 13:30–) — encoder generality

## HuBERT replication of the 3×3 — pig→cat does NOT replicate

Same 5,031 pig calls, same cats and dogs, HuBERT-base instead of WavLM (mean over layers, group perm):

| cell | WavLM | HuBERT |
|---|---|---|
| dog→cat | 0.583 (p=.015) | **0.561 (p=.060)** — holds, borderline |
| pig→cat | 0.591 (p=.005) | **0.530 (p=.229)** — does not replicate |
| cat→pig | 0.524 (p=.10) | 0.523 (p=.14) |
| pig→dog | 0.523 (p=.25) | 0.493 (p=.55) |
| pig within, held-out lab (L0) | 0.660 | 0.628 |

So: **dog→cat is the only cell that holds across two encoders** (WavLM p=.015/.0045 animal-level; HuBERT
p=.06/.037 selection-free earlier). pig→cat is a WavLM-only result and must be reported as such.
The transfer story is now: *one robust cell, one encoder-dependent cell, everything else at chance.*

Running: AVES-bio (a **bioacoustic**, non-speech encoder) on the full 3×3 — the third encoder family.
If dog→cat holds there too, it is a property of frozen encoders generally; if not, it is a speech-model
property (which would itself be interesting — speech pretraining as the carrier).

## eGeMAPS added to Paper A (see above): honest handcrafted baseline 0.486; encoders +0.07–0.09 over it.

## AVES-bio (third encoder family — bioacoustic, not speech) — the 3×3

| cell | WavLM | HuBERT | AVES-bio |
|---|---|---|---|
| dog→cat | **0.583 (p=.015)** | 0.561 (p=.060) | 0.537 (p=.23) |
| pig→cat | **0.591 (p=.005)** | 0.530 (p=.23) | **0.608 (p=.040)** |
| cat→pig | 0.524 (p=.10) | 0.523 (p=.14) | **0.559 (p=.040)** |
| pig→dog | 0.523 (p=.25) | 0.493 (p=.55) | 0.565 (p=.085) |
| cat→dog | 0.527 (p=.59) | — | 0.456 (p=.78) |
| dog→pig | 0.495 (p=.56) | — | 0.469 (p=.82) |
| placebo (sex→affect) | 0.51 | — | 0.487 |
| pig within, held-out lab | 0.660 | 0.628 | **0.679** |

**Reading.** No cell is significant in all three encoders. **pig→cat is significant in two of three**
(WavLM, AVES); dog→cat in one plus a borderline. Every encoder finds at least one species pair that
transfers *into cats* above chance, but *which* pair differs. AVES — a bioacoustic model — is the only
one where anything transfers *out* of cats (cat→pig 0.559). Placebos stay at chance in all.

Honest Paper-B sentence, final form:
> *Across three frozen encoder families, an affect axis learned on one species predicts cat contexts
> above chance in every encoder, but the specific source species that transfers is encoder-dependent
> and no cell is unanimous; the effect is small, placebo-clean, roughly half explained by call duration,
> and should be read as "encoders share some coarse arousal structure across mammals" rather than as
> a stable cross-species code.*

Figure: `figures/encoders3.png`. Data: `results/out_pigs/aves_matrix.json`, `encoder_consistency.json`.
