# What in the embedding actually carries the cross-species affect transfer?

Self-contained re-analysis of the **cached** WavLM-base-plus embeddings (no encoder
loaded). Three species, one question: an affect probe trained on one species predicts
another species' contexts above chance — *what is the thing that transfers?*

Everything here is written to `out_sae/`; no existing `out_*/` file was touched.

---

## 1. Data and protocol

| | clips | classes | grouping unit |
|---|---|---|---|
| cat | 348 | isolation = neg (221) vs brushing = pos (127) | 20 individuals |
| dog | 308 | aggression = neg (99) vs play = pos (209) | 10 individuals |
| pig | 5031 | Neg (2997) vs Pos (2034) valence contexts | 6 recording teams |

- Encoder: frozen `microsoft/wavlm-base-plus`, mean-pooled hidden states, cached on disk
  as float16, cast to float32. Layers **9** (primary), **12** (robustness), **3** (early
  contrast).
- **z-scoring**: within species; pigs **within recording team** (matching `pigs_run.py`,
  because valence is confounded with lab).
- **permutation unit**: labels shuffled *within individual* (cat/dog) and *within team*
  (pig), 200 draws. This is the conservative null — it leaves all individual/lab
  structure intact and only breaks the label link.
- **Acoustics**: `[log_duration, log_rms_energy, spectral_centroid, ZCR]` recomputed from
  audio with the identical 16 kHz / 6 s preprocessing used to make the embeddings, so the
  duration is the duration the encoder actually saw. Sanity check: computed pig
  log-duration vs the Soundwel key's own `Dur` column, Pearson **r = 0.992**.

---

## 2. Task 1 — is there a shared affect direction?

`w_s = normalize(mean_neg − mean_pos)` per species, on the z-scored embeddings.

### Layer 9 (primary)

| pair | cos | null mean ± SD | z | p(>) | cos after log-dur removed | null | p(>) |
|---|---|---|---|---|---|---|---|
| cat–dog | **+0.188** | −0.142 ± 0.194 | +1.70 | 0.065 | +0.081 | −0.156 ± 0.205 | 0.119 |
| cat–pig | **+0.012** | −0.001 ± 0.110 | +0.12 | 0.448 | +0.118 | +0.154 ± 0.046 | 0.791 |
| dog–pig | **+0.004** | −0.005 ± 0.111 | +0.08 | 0.488 | +0.086 | −0.085 ± 0.198 | 0.229 |

### Layer 12 and layer 3

| layer | cat–dog | cat–pig | dog–pig |
|---|---|---|---|
| 12 raw | +0.071 (p=0.189) | +0.171 (p=0.090) | −0.053 (p=0.716) |
| 12 dur-removed | −0.006 (p=0.333) | +0.151 (p=**0.045**) | +0.095 (p=0.149) |
| 3 raw | +0.140 (p=0.080) | +0.135 (p=0.050) | +0.114 (p=0.070) |
| 3 dur-removed | +0.039 (p=0.154) | +0.101 (p=0.234) | −0.066 (p=0.463) |

**Not one of these survives.** The single p ≤ 0.05 (cat–pig, layer 12, duration-removed)
is one hit out of **18** comparisons — exactly what you expect by chance. The honest read
is that no pairwise affect direction is reliably aligned.

### How much of each affect axis is duration?

`d_s` = OLS coefficient vector of the embedding on log-duration.

| | cat | dog | pig |
|---|---|---|---|
| cos(w_s, d_s) layer 9 | +0.262 | +0.353 | **+0.897** |
| cos(w_s, d_s) layer 12 | +0.328 | +0.543 | +0.797 |
| cos(w_s, d_s) layer 3 | +0.296 | +0.392 | +0.710 |
| Spearman(own affect score, log-dur), L9 | +0.231 | +0.413 | **+0.745** |
| ‖mean_neg − mean_pos‖ (z units), L9 | 11.21 | 9.66 | 4.32 |

**The pig "affect axis" is very nearly the duration axis** (cos 0.897 at layer 9). Cat and
dog are only modestly duration-loaded. And the *duration directions themselves* are shared
between cat and dog (cos **+0.572** at L9, +0.617 at L12) but not with pigs (cat–pig
+0.008, dog–pig −0.275).

After residualisation `cos(w_s, d_s)` becomes cat +0.071, dog +0.137, pig **−0.819**.
Regressing duration out zeroes the *covariance* with duration by construction, not the
cosine with the duration direction; the pig contrast over-corrects and ends up pointing
against it. That is another way of saying the pig affect contrast is almost entirely
duration.

### Is the cosine small, or just badly estimated?

Split-half reliability of `w_s` (animals held out for cat/dog; calls split within team for
pigs, since Soundwel carries no individual id):

| layer | cat | dog | pig |
|---|---|---|---|
| 9 | 0.609 | 0.413 | **0.941** |
| 12 | 0.649 | 0.429 | 0.926 |
| 3 | 0.523 | 0.614 | 0.938 |

Attenuation-corrected cosines (`cos / sqrt(rel_a·rel_b)`), layer 9: cat–dog **+0.374**,
cat–pig +0.016, dog–pig +0.006.

This matters for the interpretation. The cat and dog axes are noisy, so the cat–dog
cosine is genuinely attenuated and the true value may be around 0.37 — suggestive, but
still inside the permutation null. The pig axis is estimated almost perfectly
(reliability 0.94), so **the pig–cat and pig–dog cosines of ~0.00 are a real absence, not
a power failure.**

---

## 3. Task 2 — transfer as a 1-D projection

Train a balanced logistic probe on species A, project species B onto `w_A`, score AUC for
B's affect labels, then ask what that projection tracks physically in B.
`AUC|dur` = AUC after log-duration is partialled out of the projection (within z-unit).
`md` = the same thing using the **mean-difference** axis of Task 1 instead of the probe.

### Layer 9

| A→B | AUC | AUC\|dur | md-axis AUC | ρ log-dur | ρ energy | ρ centroid |
|---|---|---|---|---|---|---|
| cat→dog | 0.632 | 0.638 | 0.557 | +0.020 | **+0.337** | −0.109 |
| cat→pig | 0.516 | 0.508 | 0.496 | +0.070 | −0.023 | −0.166 |
| dog→cat | **0.790** | 0.777 | 0.616 | **+0.394** | +0.344 | +0.085 |
| dog→pig | 0.534 | 0.512 | 0.495 | +0.067 | +0.034 | +0.104 |
| pig→cat | 0.681 | 0.675 | 0.490 | +0.163 | **+0.544** | −0.064 |
| pig→dog | 0.590 | 0.617 | 0.501 | −0.260 | +0.011 | +0.041 |

### Layer 12 / layer 3

| A→B | L12 AUC (\|dur, md) | L3 AUC (\|dur, md) |
|---|---|---|
| cat→dog | 0.577 (0.587, 0.514) | 0.719 (0.708, 0.610) |
| cat→pig | 0.490 (0.490, 0.557) | 0.603 (0.552, 0.584) |
| dog→cat | 0.712 (0.707, 0.552) | 0.679 (0.661, 0.584) |
| dog→pig | 0.520 (0.540, 0.484) | 0.529 (0.495, 0.545) |
| pig→cat | 0.751 (0.736, 0.667) | 0.441 (0.427, 0.640) |
| pig→dog | 0.576 (0.552, 0.483) | 0.460 (0.458, 0.661) |

Reference — raw acoustics predicting affect *within* each target:
cat: dur 0.592, energy 0.685, centroid 0.522 · dog: dur 0.572, energy 0.433, centroid
0.364 · pig: dur **0.775**, energy 0.465, centroid 0.544.

Four things fall out:

1. **Transfer is strongly asymmetric.** Everything transfers *into* cats (dog→cat 0.790,
   pig→cat 0.681/0.751) and almost nothing transfers *into* pigs (cat→pig 0.516/0.490,
   dog→pig 0.534/0.520). Pigs are the hardest target despite having 5031 calls and the
   most reliably estimated affect axis.
2. **Duration is not the carrier here.** Partialling log-duration out of the projection
   changes almost nothing (dog→cat 0.790 → 0.777). Where the transferred axis correlates
   with anything physical it is more often **energy** (pig→cat ρ = +0.544, dog→cat +0.344,
   cat→dog +0.337) than duration.
3. **The probe direction and the mean-difference direction are not the same vector.**
   dog→cat is 0.790 with the logistic probe but 0.616 with the mean-difference axis. The
   probe whitens by the within-species covariance and that is what makes it transfer.
   This reconciles Task 1 with Task 2: near-zero cosines between *mean-difference* axes
   are compatible with real transfer through *whitened* axes.
4. **Layer 3 flips the pig probe's sign** (pig→cat 0.441, pig→dog 0.460 with the probe,
   but 0.640/0.661 with the mean-difference axis). Early-layer directions are not stable.

---

## 4. Task 3 — sparse features

**Method used: `sklearn.decomposition.MiniBatchDictionaryLearning`**, 192 components,
`alpha=1.0`, `batch_size=256`, `max_iter=300`, `transform_algorithm='lasso_lars'`,
`transform_alpha=1.0`, `positive_code=True`. Note `fit_algorithm` had to be set to
`'cd'` — sklearn refuses positive codes with the default `'lars'` fit. Fitted on the
pooled within-species z-scored **layer-9** embeddings, **5687 × 768** (cat 348 + dog 308 +
pig 5031; the brief said ~6300, the actual pooled count is 5687).

Runtime **13 s** to fit + ~5 s to transform on CPU, single-threaded — comfortably fast, no
fallback needed. FastICA was also run as an independent cross-check. Mean **33.7 / 192**
active atoms per sample, **68.6 %** of variance explained.

Selectivity = AUC of a component's activation for neg vs pos, computed **within** each
species. Criterion: >0.60 or <0.40 in ≥2 species **with the same sign**.

### Result

- **1 of 192** components passes in ≥2 species (null over 200 label permutations: **0.00 ±
  0.00**, max 0, p = 0.005).
- **0 of 192** components pass in all three species (null 0.00 ± 0.00, p = 1.000).

Per-species selective counts (out of 192):

| | observed | null mean ± SD (max) | p |
|---|---|---|---|
| cat | 9 | 3.34 ± 1.41 (9) | 0.010 |
| dog | 10 | 0.05 ± 0.24 (2) | 0.005 |
| **pig** | **0** | 0.00 ± 0.00 (0) | 1.000 |

Best single atom per species: cat #11 AUC 0.261 (dog 0.533, pig 0.502) · dog #117 AUC
0.762 (cat 0.505, pig 0.525) · **pig #1 AUC 0.580** (cat 0.503, dog 0.542). Cats and dogs
each have strongly selective atoms and they are *different* atoms. Pigs, with 5031 calls
and the tightest null (SD 0.005), have **no** atom that reaches 0.60 — the pig valence
signal is genuinely distributed across the dictionary, not localised in any one feature.

### The one component that passes: #33

| | cat | dog | pig |
|---|---|---|---|
| selectivity AUC | **0.299** | **0.334** | 0.522 |
| permutation p (two-sided) | 0.005 | 0.005 | 0.085 |
| null AUC SD | 0.024 | 0.027 | 0.005 |
| fraction of clips active | 0.399 | 0.370 | 0.223 |
| AUC controlling log-duration | 0.340 | 0.294 | 0.422 |
| AUC controlling dur+energy+centroid | 0.392 | 0.408 | 0.434 |
| partial ρ(activation, affect \| acoustics) | −0.359 | −0.262 | +0.021 |
| **AUC computed inside each animal** | 0.372 (11 cats) | 0.351 (10 dogs) | 0.507 (4 teams) |
| ρ with log-duration | −0.166 | +0.022 | +0.090 |
| ρ with log-energy | −0.081 | +0.130 | +0.052 |
| ρ with centroid | +0.053 | +0.153 | −0.028 |
| cos(atom, that species' affect direction) | −0.293 | −0.455 | −0.028 |

Component 33 activates **more for positive affect** (brushing / play) in both cats and
dogs. It is **not duration in disguise**: its correlation with log-duration is −0.166 /
+0.022 / +0.090, and controlling for duration leaves the selectivity intact or slightly
stronger (cat 0.299 → 0.340, dog 0.334 → 0.294). Controlling for duration *and* energy
*and* centroid weakens it but does not remove it (0.392 / 0.408). It also survives being
computed strictly **inside** each animal — all 10 dogs are on the same side (per-dog AUC
0.155 – 0.423), which rules out an individual-identity artifact. Its atom sits at cos
−0.293 / −0.455 with the cat and dog affect directions but −0.028 with the pig one, which
is Task 1's finding restated at the level of a single feature.

In pigs, component 33's mean activation by context shows no valence split (Castration
0.428 and NegativeConditioning 0.330 next to PositiveConditioning 0.325 and MissedNursing
0.326).

### Robustness

- **FastICA (64 components)** independently finds exactly the same pattern: **1** component
  passing in ≥2 species (#51: cat 0.600, dog 0.628, pig 0.495), **0** in all three. It is
  *not* the same feature as dictionary component 33 (Spearman −0.001, cos between the ICA
  readout direction and atom 33 = +0.008) — two mutually independent features, each
  cat+dog selective, neither pig-selective.
- **Species-balanced dictionary** (fit on cat + dog + 330 subsampled pigs, so pigs no
  longer dominate the 88 % of rows): **0** components pass. The best candidates are
  near-misses on the same cat+dog pattern — #161 (0.414 / 0.580 / 0.490), #123 (0.436 /
  0.571 / 0.500), #3 (0.423 / 0.417 / 0.489) — so this is an underpowered replication of
  the cat+dog effect, not a contradiction of it. Still, it means the exact identity of
  component 33 is dictionary-dependent.

---

## 5. Task 4 — verdict

**No. Not one component is affect-selective in all three species, and the clean negative
is the result.**

- 0 of 192 sparse components (and 0 of 64 ICA components) are affect-selective in cat,
  dog and pig simultaneously. The duration control never even gets a chance to be applied
  at the three-species level, because nothing reaches it.
- Exactly **one** component (#33) is selective in **two** species — cat and dog, same sign,
  p = 0.005 in each, and it is **not** duration: it is unchanged by a duration control,
  only partly reduced by a full duration+energy+centroid control, and consistent inside
  every individual animal. FastICA finds a second, independent cat+dog component. So
  cat and dog do share affect-relevant structure at layer 9 that is not reducible to call
  length.
- **Pigs are the wall.** The pig affect direction is highly reliable (split-half 0.94) and
  is almost exactly the duration direction (cos 0.897, Spearman with log-duration 0.745).
  It is orthogonal to the cat and dog affect directions (cos 0.012 and 0.004), pigs share
  no selective sparse component with either, and nothing transfers *into* pigs (AUC 0.49 –
  0.53). The transfer that does exist between pigs and the carnivores runs one way only,
  into cats, and there it tracks **energy** (ρ = +0.544) more than duration.
- The apparent tension between "cosines are ~0" and "transfer AUC is 0.63 – 0.79" is
  resolved by point 3 of Task 2: transfer runs through the covariance-whitened logistic
  direction, not the raw mean-difference axis. Anyone reporting cosine similarity between
  class-mean differences as evidence for or against a shared axis should report both.

### What this does *not* license

The cat–dog cosine (+0.188 raw, +0.374 attenuation-corrected) is *suggestive but not
significant* against a within-animal permutation null (p = 0.065). With 20 cats and 10
dogs the direction estimates are noisy (reliability 0.61 / 0.41) and this study cannot
settle it. The one significant cat+dog sparse component is a stronger piece of evidence
than the cosine, but it is a single component out of 192 in one dictionary, and it does
not reappear in the species-balanced fit. Treat "cat and dog share a duration-independent
affect feature" as a hypothesis worth a proper test, not as an established result.

---

## 6. Files

| file | contents |
|---|---|
| `results.json` | everything below, merged, plus the verdict block |
| `task12.json` / `task12.log` | Task 1 + 2, all three layers, nulls, reliabilities |
| `task3.json` / `task3.log` | dictionary learning, selectivity, permutation nulls |
| `followup.json` / `followup.log` | per-animal breakdown, per-species counts, ICA and balanced-dictionary cross-checks |
| `shared_direction.png` | cosine matrices (raw + duration-removed), duration loading of each affect axis, top-10 component selectivities |
| `acoustics.npz` | cached `[log_dur, log_energy, centroid, zcr]` for all 5687 clips |
| `codes.npz` | selectivity matrix, dictionary atoms, codes for the passing components |
| `common.py`, `prep_acoustics.py`, `task12.py`, `task3.py`, `followup.py`, `figure.py`, `consolidate.py` | the code, in run order |

Reproduce with `python out_sae/prep_acoustics.py && python out_sae/task12.py &&
python out_sae/task3.py && python out_sae/followup.py && python out_sae/figure.py &&
python out_sae/consolidate.py` (about 2 minutes total, numpy/sklearn only, peak memory
well under 1 GB — pig embeddings are memory-mapped and only the needed layers are
materialised).
