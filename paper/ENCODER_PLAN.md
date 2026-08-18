# Building our own representation — the plan

*17 Aug 2026. Written in response to: "what are the important features of animal audio, what are
existing models missing, and could we build a better one?"*

---

## 0. The honest framing first

There are two different things "build our own model" could mean, and only one of them is winnable.

**Not winnable: a new foundation encoder.** Perch 2.0 trained on 1.5M recordings; AVEX swept 26
datasets; NatureLM used ~22,000 hours. They have TPU pods. We have an M2 with 8 GB and no GPU. If we
train a competitor from scratch we produce a worse Perch. That door is closed and we should say so
plainly rather than discover it in three months.

**Winnable, and pointed at by our own measurements: the layer that sits on top.** Every result we
produced this week says the same thing — *the information is in there, but it is dominated by things
we do not want.* Identity decodes at 0.63–0.94 while context sits at 0.55–0.73. Random splits inflate
by 10–24 points because of it. 88 hand-crafted features from 2015 **match** a 95M-parameter
self-supervised encoder once you evaluate honestly.

That last fact is the tell. If a 768-dimensional learned representation cannot beat 88 numbers on the
task we care about, the bottleneck is not capacity. **It is that nobody has built the representation
for this problem.** They built it for speech, and we are borrowing it.

So the goal is not a bigger encoder. It is: **given frozen encoders as a free input, construct a
representation that keeps context and discards who/where.** That is cheap, it is measurable with the
harness we already have, and as far as our novelty checks show, nobody has done it in bioacoustics.

---

## 1. The feature space, laid out properly

Think of it as four layers. Almost all published work varies layer 3 and ignores the rest.

### Layer 1 — the frontend (waveform → time-frequency)

| representation | what it buys | who uses it |
|---|---|---|
| raw waveform | nothing thrown away | wav2vec2, AVES (conv frontend) |
| STFT / linear spectrogram | honest frequency axis | classical DSP |
| **log-mel** | perceptual compression — **calibrated to human hearing** | BEATs, Perch, BirdNET, ~everything |
| **PCEN** | adaptive gain control; robust to channel and distance | bird detection literature |
| CQT | log-frequency, good over wide ranges | music, some bioacoustics |
| gammatone | auditory-filter model | psychoacoustics |
| wavelet scattering | translation-invariant, deformation-stable | Andén & Mallat |
| learnable (LEAF, SincNet) | fit the frontend to the task | mixed evidence |

**The gap that stands out.** The mel scale is a fit to *human* pitch perception. A cat's audiogram
peaks around 8 kHz, a dog's extends past 40 kHz, an Egyptian fruit bat's calls run to 100 kHz+. Using
mel for all of them is a human-centric prior nobody in this field seems to question. The Greenwood
function gives cochlear frequency-position maps per species — a **species-calibrated filterbank** is
constructible from published audiograms. *(Novelty check running.)*

**And we already have evidence the frontend matters:** folding ultrasonic bat content into band via
4× time-expansion moved identity from 0.631 to 0.700. That is a frontend change, not a model change.

### Layer 2 — hand-crafted descriptors (interpretable, and stronger than expected)

eGeMAPS-88 is the standard: F0, jitter, shimmer, HNR, formants, spectral slope, loudness, MFCC stats.
**On our pig task it matched WavLM under honest evaluation.** But it was designed for the human voice,
and three things it lacks look important for animals:

1. **Nonlinear phenomena** — deterministic chaos, subharmonics, biphonation, frequency jumps. These
   are extensively documented in mammal distress and high-arousal calls, and they are in *no*
   standard feature set and no neural encoder's explicit design. If arousal lives anywhere obvious,
   it lives here. *(Suspected gap; verification running.)*
2. **Sequence structure** — inter-call interval, call rate, rhythm, bout shape. We found duration is
   the one cue that points the same way in cats and dogs. Duration is the crudest possible temporal
   feature; rate and rhythm are unexplored.
3. **Body-size normalisation** — see below, this is our best idea.

### Layer 3 — neural encoders (where everyone plays)

Speech SSL (wav2vec2/HuBERT/WavLM), general audio (BEATs/EAT), bioacoustic (AVES/AVEX/Perch/BirdNET),
audio-text (CLAP/BioLingual/NatureLM). We have measured five of these. Their honest scores cluster
within ~3 points of each other and within ~8 points of eGeMAPS. **Swapping encoders is not where the
gains are.**

### Layer 4 — metadata (almost entirely unused, and free)

You raised this and I think it is the most under-exploited layer:

- **taxonomy** — species, genus, family; phylogenetic distance between species
- **individual** — ID, sex, age
- **body mass** ← the interesting one
- **geography** — lat/long, habitat
- **time** — hour, season, sunrise offset
- **equipment** — recorder model, sample rate, gain

Only a couple of systems use any of this (spatiotemporal priors for bird ID). Nobody uses body mass.

---

## 2. ~~The idea I think is genuinely ours~~ — TESTED 17 Aug, IT DOES NOT WORK

> **Result:** allometric pitch normalisation gives 0.578 mean cross-species transfer against 0.572 for
> raw pitch, 0.569 for within-species z-scoring, and **0.577 for using no pitch feature at all**
> (+0.003, p=0.69). Dead.
>
> **Why, and it is a conceptual error not a data problem:** the pitch→affect weight is cat −0.20,
> dog −1.24, **pig +0.21**. The sign flips. Allometric normalisation shifts and rescales the pitch
> axis; it cannot reverse it. I proposed this *because* cues point in opposite directions across
> species, then proposed a fix incapable of addressing opposite directions.
>
> **Salvage:** within the 10 dogs with real measured masses, log F0 scales at b=−0.334, R²=0.48,
> p=0.026 — an independent replication of the carnivore/isometric exponent on a 2008 corpus collected
> for another purpose. And the surviving cross-species fact is cleaner than the hypothesis was:
> **duration is the only affect cue with a consistent sign across cats, dogs and pigs.**
>
> Full detail in FINDINGS.md iteration 14. Original text below, kept for the record.

## 2. (original) The idea I think is genuinely ours

**Allometric normalisation of pitch.**

Fundamental frequency scales with body size across mammals — bigger animal, longer vocal folds, lower
voice. That is physics, and the scaling law is published (Fitch; Bowling et al. on primates and
carnivores). A 3 kg cat and a 300 kg pig cannot be compared on raw F0 at all.

Now recall our own puzzle: **spectral centroid separated affect in *opposite directions* in cats and
dogs**, and pig affect turned out to be almost entirely duration. We treated that as a curiosity. But
what if the cross-species signal is not raw pitch — it is **pitch relative to what that animal's body
predicts**? An animal calling *higher than its size predicts* is straining, in any species.

That gives a concrete, testable feature: `F0_observed / F0_predicted(body_mass)`. Same for formant
dispersion, which is an even better size cue.

If that single normalised feature transfers across species where raw pitch does not, it is a clean,
interpretable, publishable result — and it comes straight out of a discrepancy in our own data. This
is the first thing I would test.

---

## 3. What to actually build: an identity-invariant projection

Our measurements say the enemy is nuisance dominance. So build the thing that removes it.

**Input:** cached frozen embeddings (we have them) + eGeMAPS + metadata.
**Output:** a representation where identity is *not* linearly decodable and context still is.
**Cost:** numpy on a laptop. No GPU. No retraining.

Candidate methods, cheapest first:
1. **Linear nullspace projection** — estimate the identity subspace, project it out. One afternoon.
   (INLP / LEACE are the modern named versions; WCCN and NAP from speaker verification are the
   1990s–2000s ancestors and are directly on point.)
2. **Within-class covariance normalisation** — whiten by within-individual covariance. Note we already
   found transfer runs through the *whitened* direction, which is a strong hint this family is right.
3. **LDA-style objective** — maximise between-context over within-individual scatter.
4. Adversarial (gradient reversal) — needs training, only if the linear methods plateau.

**The evaluation trap to avoid:** you can make identity undecodable by destroying the embedding.
So every result must report *both* numbers — identity probe accuracy **and** context accuracy — plus
the random-vs-honest gap. If identity drops and context holds, we won. If both drop, we compressed.

---

## 4. The plan, in order

### Phase 1 — cheap wins on cached embeddings ✅ **RUN 17 Aug — see FINDINGS.md iteration 12**

**Result: the premise holds for individuals, fails for sites.** Removing 32 identity directions drops
individual identity by 0.47 (cats) and 0.55 (dogs) with context unchanged or marginally better; the
same rank of *random* directions removes nothing, so this is invariance and not compression. But pig
*lab* identity falls only 0.11 from 0.941 — site identity is not a low-rank subspace. Fusion
(eGeMAPS+WavLM) gives +0.014 on pigs. Original plan below.

### Phase 1 (original) — cheap wins on cached embeddings
- **1a. Fusion.** Does eGeMAPS + WavLM beat either alone? They match each other; if they are making
  *different* errors, concatenating should help. Never tested. One hour.
- **1b. Identity-subspace removal.** ✅ done — the prediction was right for individuals, wrong for labs.
  That asymmetry is now the most interesting thing in the plan and reframes Phase 4 (below).
- **1c. Metadata fusion.** Add sex, age, breed/lab as explicit features. Cheap, and it tells us how
  much the encoder is already carrying implicitly.

### Phase 2 — frontends (next, ~days)
- **2a. PCEN vs log-mel** under the channel stress test we already built. We have the harness; this
  is a direct measurement of the invariance claim.
- **2b. Species-calibrated filterbank** from published audiograms via the Greenwood function.
  Compare against mel on the same tasks.
- **2c. Nonlinear-phenomena features** — implement chaos/subharmonic detection, add to eGeMAPS,
  test on the affect tasks.

### Phase 3 — the allometry idea ❌ **RUN AND REJECTED 17 Aug — see above**
- **3a.** Get body masses per species (and per individual where available — the dog corpus has weights).
- **3b.** Fit the F0–mass scaling law on our data; compute normalised pitch.
- **3c.** Re-run cross-species transfer with allometric features. **Prediction: it improves
  transfer where raw pitch fails.** If wrong, that is a clean negative and still worth writing.

### Phase 4 — assemble and name it (~weeks)
A small package: frozen encoder → nuisance projection → fused descriptors + metadata → probe. Ship it
with the honest evaluation protocol built in, so the thing that makes it useful is that it *reports
its own leakage*. That is a tool people would actually install.

---

## 5. What I would not do

- Train a foundation encoder. Explained above.
- Chase a bigger encoder. Their honest scores are within 3 points of each other.
- Add capacity before removing nuisance. The measurement says nuisance is the bottleneck.
- Any of this before the ICBINB deadline on **29 August**. Paper A is written and 12 days out; this
  plan is what comes *after* it. Phase 1a/1b are cheap enough to run alongside, and if the
  identity-removal result is strong it becomes a natural follow-up paper rather than a rushed section.

---

## 5b. What Phase 1 changed

The plan assumed "remove nuisance" was one problem. It is two:

- **individual nuisance** — low-rank, linearly removable, free. A shippable tool.
- **site/channel nuisance** — high-rank, distributed, *not* linearly removable. Needs a nonlinear or
  adversarial method, i.e. training, i.e. the GPU we do not have yet.

Phase 4 should therefore ship the individual-invariant projection now, and treat site invariance as
the open research problem — which is also the one that matters most for real deployment, since
ecologists move recorders between sites far more often than they change animals.

## 6. Open questions the research agents are checking
- Has anyone built a species-calibrated (non-mel) filterbank?
- Is allometric normalisation used as a cross-species *feature* anywhere?
- Are nonlinear vocal phenomena in any standard descriptor set?
- Has identity/site-invariance been attempted in bioacoustics at all?
- What is the standard protocol for proving invariance without compression?
