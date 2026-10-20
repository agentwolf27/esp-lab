# Does normalising pitch by body size make animal calls comparable across species?

**Verdict: No.** Allometric normalisation of F0 does not improve cross-species affect
transfer over raw F0 (mean Δ = **+0.009** balanced accuracy over 6 directed pairs,
sign test p = 0.22), does not beat within-species z-scoring (Δ = **+0.011**, p = 0.22),
and — decisively — does not beat **deleting the pitch channel entirely**
(Δ = **+0.003**, p = 0.69). Every variant lands on the same number, ~0.57.

The idea is not refuted because the allometric law is false. Within our dogs, where we
have real measured masses, the law holds beautifully (b = −0.334, R² = 0.48, p = 0.026 —
textbook isometric −1/3). The idea fails because **pitch carries almost no
species-transferable affect signal to begin with**, and because the pitch→affect relation
*flips sign* between pigs and cats/dogs. A location/scale correction cannot repair a
sign flip.

![results](allometry.png)

---

## 1. What was run

| stage | choice |
|---|---|
| unit of analysis | the **focal call** — loudest contiguous energetic segment per file, not the file |
| F0 | `librosa.pyin`, per-species range, frame ≥ 4 periods of `fmin`, resolution 0.15 semitone |
| body-size cue | F0 only. Formant dispersion was attempted and **could not be measured** (§4) |
| features | pitch channel + log₁₀(call duration) + log₁₀(RMS) |
| classifier | logistic regression, `class_weight='balanced'`, scaler fitted on the **training species only** |
| metric | balanced accuracy |
| null | training labels permuted **within individual (cats, dogs) / lab (pigs)**, 200 draws |
| affect axis | NEGATIVE = cat isolation / dog aggression / pig Neg valence; POSITIVE = cat brushing / dog play / pig Pos |

Pigs were subsampled to ≤150 calls per (recording team × context category) to keep the
laptop honest: 6 887 → 2 729 affect-labelled calls.

### Why the focal-call segmentation matters

Cat files are single meows (median 1.84 s), pig files are single calls (0.30 s), but dog
files are **bout recordings** (median 12.2 s, max 102 s). Comparing raw file duration
across those compares segmentation conventions, not animals. Extracting the focal call
fixes this, and it is validated: our `dur_call` correlates **r = 0.976** with the
Soundwel corpus's own expert-measured `Dur`. It also matters empirically — negative-minus-positive
separation on log duration rises from +0.08 (cat) / +0.50 (dog) with *file* duration to
**+0.29 / +1.18** with *call* duration.

---

## 2. Body-mass assumptions

| species | source | value(s) | quality |
|---|---|---|---|
| **dog** | `annotations.csv`, per individual | 6, 16, 18, 19, 25, 25, 32, 34, 34, 36 kg | **measured, per animal** — the only real masses we have |
| **cat** | no mass in CatMeows. Imputed from breed × sex | EU♀ 4.0, EU♂ 5.0, MC♀ 5.5, MC♂ 8.0 kg | **imputed**, 4 distinct values for 21 cats; neuter status ignored (9/21 cats are EU♀) |
| **pig** | no mass in the Soundwel key. Imputed from Age Category, midpoints of standard production stages | Piglet 5, Weaner 20, GrFinishing 75 kg | **imputed**, 3 distinct values for 1 260 calls |

Pig stage masses assume: suckling piglet ≈ 1.5–9 kg; weaner from weaning (3–5 wk) to
≈ 20 kg at ≈ 8 wk; grower–finisher 20 kg → ≈ 110 kg slaughter at 5–6 months.

**Three confounds that must be stated:**

1. **Pig mass is perfectly collinear with recording lab.** IASPA/B/C recorded only
   piglets, ETHZ/FBN only weaners, NMBU only grower-finishers. "Mass" and "lab" are the
   same variable in the pig corpus. Anything mass appears to explain in pigs may be site.
2. **Pig mass correlates with valence** (Piglet 2446 Neg/1123 Pos; Weaner 1541/359;
   GrFinishing 585/833), so a mass-based correction is not label-neutral within pigs.
3. **Cat and pig "mass" have 4 and 3 levels.** Within those species the allometric
   residual is log F0 minus a near-constant, i.e. essentially species-mean centring (§6).

---

## 3. F0 extraction and its honest failure rate

Ranges used, chosen from literature and checked against the corpora:

| species | fmin–fmax | justification |
|---|---|---|
| cat | **180–1500 Hz** | meow F0 typically 350–800 Hz (Nicastro 2004; Schötz 2019). Cat audio is natively 8 kHz |
| dog | **110–2000 Hz** | bark F0 typically 250–900 Hz, growls far lower (Yin & McCowan 2004) |
| pig | **40–1500 Hz** | the task brief suggested 100–1000 Hz; **that floor is wrong**. The Soundwel key's own `F0Mean` has median 59 Hz and minimum 25 Hz — adult grunts sit at 50–80 Hz |

| species | affect clips | F0 succeeded | on POSITIVE | on NEGATIVE | analysed |
|---|---|---|---|---|---|
| cat | 348 | 95.7 % | 88.2 % | **100 %** | 333 |
| dog | 308 | 75.6 % | 77.5 % | 71.7 % | 233 |
| pig | 2 729 | 46.2 % | 40.2 % | **49.6 %** | 1 260 |

**F0 extraction failure is not missing-at-random: it correlates with the affect label**
in every species, most strongly in cats (100 % vs 88 %) and pigs (50 % vs 40 %).
Voiceless/atonal calls are disproportionately *positive*. Discarding them — which we do,
honestly, rather than imputing — therefore shifts class balance. Balanced accuracy and
`class_weight='balanced'` mitigate but do not remove this.

### Validation against an independent measurement

On the 861 pig calls where both exist, our pyin F0 vs the corpus's published `F0Mean`
(Praat/Avisoft, tuned per corpus by the original authors): **log–log r = 0.78**, median
ratio 1.31, and **44 % of clips more than 1.6× the published value** — substantial
octave-doubling. Eight alternative pyin configurations were tried; none exceeded r = 0.735.

This is itself evidence against the premise. A single uniform pitch tracker cannot be
trusted across species whose F0 ranges differ tenfold. **Measurement non-comparability is
a confound that no post-hoc normalisation can fix**, and it sits upstream of the whole
idea. Section 8 therefore repeats everything using the corpus's own expert F0 for pigs.

---

## 4. Formant dispersion: attempted, and not measurable here

Formant dispersion is the better body-size cue (Fitch 1997; Charlton & Reby) and we tried
it: LPC over the 0–4 kHz band (the lowest common bandwidth — cats are 8 kHz-sampled),
roots-of-polynomial formant picking, Reby & McComb slope estimator. It is **degenerate**:

| LPC order | cat Df | dog Df | pig Df | formants found |
|---|---|---|---|---|
| 12 | 642 Hz | 638 Hz | 690 Hz | 6, 6, 6 |
| 8 | 946 Hz | 911 Hz | 1021 Hz | 4, 4, 4 |

Df ≈ 4000/n_formants in every cell: the estimator returns the **LPC model order**, not
the animal. Tightening bandwidth and pole-magnitude criteria made it worse and ranked
pigs — the largest animals — as having the *highest* dispersion, backwards from body size.

The reason is physical, not a coding bug. A cat's vocal tract (≈7 cm) has formant spacing
c/2L ≈ 2 460 Hz, so in 0–4 kHz there are barely two formants; meanwhile cat F0 is ≈ 600 Hz,
so the spectrum is sampled only every 600 Hz. You cannot resolve a 2 460 Hz envelope
period from a comb with 600 Hz teeth inside a 4 kHz band. **Formant dispersion is not
recoverable from this material** and was excluded rather than reported as a feature.

---

## 5. The fitted allometric law

Regressing log₁₀ F0 on log₁₀ mass:

| fit | n | exponent b | R² | p |
|---|---|---|---|---|
| **group level (17 mass groups) — primary** | 17 | **−0.389** | 0.205 | **0.068 (n.s.)** |
| clip level, species-balanced weights | 1 826 | −0.379 | 0.125 | — |
| clip level, unweighted | 1 826 | −0.660 | 0.259 | <1e-100 |
| species level (3 points) | 3 | −0.475 | 0.233 | 0.68 |
| **within dog** (measured masses, 10 individuals) | 10 | **−0.334** | **0.483** | **0.026** |
| within cat (4 imputed mass groups) | 4 | +0.084 | 0.044 | 0.79 |
| within pig (3 imputed mass groups) | 3 | −0.837 | 0.723 | 0.35 |

Published references, converted to a **mass** exponent (Bowling et al. 2017 regressed on
log body **length**; assuming isometry M ∝ L³, divide by 3):

| source | exponent on log mass |
|---|---|
| isometric theory (vocal-fold length ∝ M^⅓, F0 ∝ 1/L) | −0.333 |
| Bowling et al. 2017, carnivores (β = −1.004, R² = 0.443) | −0.335 |
| Bowling et al. 2017, primates (β = −2.456, R² = 0.741) | −0.819 |

**Our −0.389 sits squarely inside the published range** — but the point estimate is the
only good news. R² = 0.21 and p = 0.068: **three species do not establish the law.**

Two things make this worse:

- **The law is not identifiable from two species.** Refitting with one species held out
  gives b = −0.561 (no cat), **−0.907** (no dog), **−0.089** (no pig). A tenfold swing in
  the exponent depending on which species you happen to have.
- **The pig points are a call-type artefact, not allometry.** Sampled 5 kg piglets have
  median F0 689 Hz; sampled 20 kg weaners have 65 Hz. That is a 10× frequency drop for a
  4× mass change — because the piglet sample is dominated by screams and the weaner
  sample by grunts. Composition of call types, not body size, is driving the between-group
  variance that the fit is reading as a law.

---

## 6. Is the allometric residual anything other than knowing the species?

`F0_residual = log F0_observed − (â + b̂ · log mass)`. Compared with simply subtracting the
species mean:

| species | correlation with species-mean-centred F0 | sd(allo − centred) vs within-species sd(log F0) | observed/predicted F0 |
|---|---|---|---|
| cat | **0.956** | 0.040 vs 0.128 | 0.77× |
| pig | **0.963** | 0.154 vs 0.541 | **0.39×** |
| dog | 0.751 | 0.104 vs 0.152 | 1.49× |

For cats and pigs the allometric residual **is** species-mean centring, to r ≈ 0.96 —
because mass takes 4 and 3 values. Only in dogs, where masses are real and per-individual,
does allometry say anything a species indicator does not. And the fitted law mispredicts
species-average F0 by up to **2.6×** (pigs at 0.39× predicted), which is a large error for
something meant to put species on a common scale.

---

## 7. The transfer test

Feature vectors are pitch + log call duration + log RMS. Duration and energy are
within-species z-scored in **all** variants so that the comparison isolates the pitch
channel; the scaler is then fitted on the training species only. `nopitch` drops the
pitch channel entirely.

| pair | no pitch | (a) raw F0 | (b) allometric | (b′) allometric, law excl. test sp. | (c) within-species z | species-mean centred |
|---|---|---|---|---|---|---|
| dog → cat | 0.479 | 0.539 | 0.567 | 0.571 | 0.519 | 0.527 |
| pig → cat | 0.527 | 0.552 | 0.525 | 0.525 | 0.500 | 0.513 |
| cat → dog | 0.519 | 0.550 | 0.511 | 0.551 | 0.550 | 0.553 |
| pig → dog | 0.747 | 0.711 | 0.733 | **0.748*** | 0.724 | 0.747 |
| cat → pig | 0.492 | 0.537 | 0.544 | 0.544 | 0.499 | 0.514 |
| dog → pig | 0.702 | 0.540 | 0.588 | 0.544 | 0.624 | 0.497 |
| **mean** | **0.577** | **0.572** | **0.578** | **0.580** | **0.569** | **0.559** |

`*` = p < 0.05 against the within-group permutation null. **Only one of six pairs
(pig → dog) is significant — and it is significant for the no-pitch baseline too**
(nopitch pig→dog: 0.747, null mean 0.688, p = 0.005; all five other pairs p ≥ 0.06).
Raw F0 reaches p < 0.05 in **zero** of six pairs (best: pig→dog, p = 0.060).

Aggregate tests over the 6 pairs:

| comparison | mean Δ | better in | sign test p | Wilcoxon p |
|---|---|---|---|---|
| allometric (b′) vs raw | +0.0089 | 5/6 | 0.22 | 0.22 |
| allometric (b′) vs within-species z | +0.0112 | 5/6 | 0.22 | 0.44 |
| allometric (b′) vs **no pitch at all** | +0.0031 | 4/6 | 0.69 | 0.69 |
| allometric (b) vs raw | +0.0063 | 4/6 | 0.69 | 0.69 |
| within-species z vs raw | −0.0023 | 3/6 | 1.00 | 0.84 |
| species-mean centred vs raw | −0.0129 | 2/6 | 0.69 | 0.31 |

Per-pair paired cluster bootstraps (1 000 draws, resampling test-side individuals/labs)
give Δ CIs that exclude zero in **both directions** — allometry is significantly *worse*
than raw for pig→cat (−0.027) and cat→dog (−0.039), and significantly *better* for
dog→pig (+0.048) and pig→dog (+0.037). That pattern is noise, not an effect.

Within-species leave-one-individual/lab-out, for reference: cat 0.629 / dog 0.805 /
pig 0.673 with raw F0, and 0.641 / 0.771 / 0.677 with the allometric residual. The probes
work within species; they do not transfer.

### The permutation nulls are not at 0.5

Null means by pair (labels shuffled within individual/lab): dog→cat 0.461, cat→pig 0.516,
dog→pig 0.514, pig→cat 0.583, cat→dog 0.522, **pig→dog 0.670**. Because several cats
(5 of 20) and two of six pig labs contribute only one class, within-group shuffling cannot
break the between-group label structure, and a "shuffled" model still scores well above
chance. **Reporting these transfer numbers against a naive 0.5 chance line would badly
overstate them** — the apparently strong pig→dog result (0.75) sits only ~0.08 above its
own null.

---

## 8. Sensitivity checks

| variant | no pitch | raw | allometric | allo. LOSO | within-sp. z | sp.-mean centred |
|---|---|---|---|---|---|---|
| main (our pyin, call duration) | 0.577 | 0.572 | 0.578 | 0.580 | 0.569 | 0.559 |
| **pigs use the corpus's published F0Mean** (n_pig = 1 636) | **0.556** | 0.512 | 0.515 | 0.516 | 0.524 | 0.522 |
| file duration instead of call duration | 0.511 | 0.534 | 0.527 | 0.536 | 0.526 | 0.515 |

The published-F0 row is the important one. With an expert-measured pig F0 the fitted law
degrades to b = −0.240, R² = 0.05, and **every pitch variant now falls below the no-pitch
baseline** — adding pitch, however normalised, actively hurts. The conclusion does not
depend on our pitch tracker's octave errors; if anything our tracker flattered the idea.

---

## 9. Why it fails

Each species' own affect probe (standardised coefficients; + = predicts NEGATIVE affect):

| species | pitch | call duration | energy |
|---|---|---|---|
| cat | **−0.198** | +0.265 | +0.707 |
| dog | **−1.240** | +1.427 | −0.133 |
| pig | **+0.208** | +1.251 | −0.307 |

Cosine between the three affect axes: cat~dog +0.36, dog~pig +0.63, **cat~pig +0.07**.

Distressed cats and dogs call *lower* than their contented counterparts; distressed pigs
call slightly *higher*. Normalising pitch by body size changes where each species' pitch
distribution sits; it cannot change which direction affect moves along it. The hypothesis
"an animal calling higher than its size predicts may be straining, in any species" requires
a shared sign, and the sign is not shared.

Only **call duration** agrees in all three species (+0.27 / +1.43 / +1.25) — and it is
duration and energy alone, with no pitch at all, that produce the 0.577 baseline every
pitch variant merely matches.

---

## 10. Verdict

- **Does allometric normalisation improve cross-species transfer over raw features?**
  No. +0.009 balanced accuracy over 6 pairs, 5/6 pairs nominally better, sign test
  p = 0.22, per-pair bootstrap CIs straddling and crossing zero in both directions.
- **Over within-species z-scoring?** No. +0.011, p = 0.22. And the cheaper control is
  itself no better than raw (−0.002), so there is nothing here for allometry to beat.
- **Is the effect even worth measuring?** No. Dropping pitch entirely scores 0.577 —
  the same as the best pitch variant. Under the published-pig-F0 sensitivity, no pitch
  (0.556) beats every pitch variant.

We should not flatter this. The one genuinely positive finding is orthogonal to the
hypothesis: **within dogs, log F0 scales with measured body mass at b = −0.334, R² = 0.48
— an independent replication of the isometric/carnivore exponent on data nobody collected
for that purpose.** That is worth reporting on its own. It is not evidence that
size-normalised pitch is a cross-species affect currency.

### What would change our mind

1. Per-individual measured masses in more than one species. Cats and pigs currently
   contribute 4 and 3 mass levels; the "law" for them is a species indicator in disguise.
2. Call-type-matched sampling. The pig piglet/weaner F0 gap is a scream/grunt composition
   artefact and it dominates the fit.
3. A species set whose pitch→affect sign is actually shared. With cat/dog negative and pig
   positive, no monotone rescaling of pitch can transfer, so this data cannot test the
   hypothesis fairly.
4. A body-size cue that survives measurement — formant dispersion is the right cue and is
   unrecoverable at these bandwidths and F0s. Wideband recordings (≥32 kHz) of a
   low-F0 species would be needed.

---

## Files

| file | contents |
|---|---|
| `extract.py` | audio → per-clip features (focal-call segmentation, pyin F0, LPC dispersion, energy, spectral shape) |
| `analyze.py` | mass assignment, allometric fits, pitch-channel construction, 6-pair transfer with permutation nulls |
| `diagnostics.py` | no-pitch baseline, paired cluster bootstraps, allometry-vs-centring, sensitivity runs |
| `aggregate.py` | sign/Wilcoxon tests over the 6 pairs, per-species probe coefficients |
| `figure.py` | builds `allometry.png` |
| `features.csv` | 3 862 clips × 34 raw features |
| `analysis_set.csv` | the 1 826 clips with complete F0/duration/energy used in the transfer test |
| `results.json` | every number quoted above |
| `extract.log` | extraction run log |

Reproduce: `OMP_NUM_THREADS=2 python extract.py && python analyze.py && python diagnostics.py && python aggregate.py && python figure.py`

### References

- Fitch, W. T. (1997). Vocal tract length and formant frequency dispersion correlate with body size in rhesus macaques. *JASA* 102(2), 1213–1222. <https://pubmed.ncbi.nlm.nih.gov/9265764/>
- Bowling, D. L. et al. (2017). Body size and vocalization in primates and carnivores. *Scientific Reports* 7, 41070. <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5259760/>
- Garcia, M. & Ravignani, A. (2020). Acoustic allometry and vocal learning in mammals. *Biology Letters* 16(7), 20200081. <https://royalsocietypublishing.org/doi/10.1098/rsbl.2020.0081>
- Briefer, E. F. et al. (2022). Classification of pig calls produced from birth to slaughter according to their emotional valence and context of production. *Scientific Reports* 12, 3409. <https://www.nature.com/articles/s41598-022-07174-8>
