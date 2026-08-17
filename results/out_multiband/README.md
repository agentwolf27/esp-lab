# Does ESP's adaptive multi-band encoding beat time-expansion on bat identity?

**Short answer: no.** On 10-way Egyptian-fruit-bat individual ID, plain 4x
time-expansion beats both multi-band fusions, for both encoders. Multi-band
concat-fusion is the better of the two fusions and does recover *some* of the
ultrasonic information, but it lands ~2.5-3 points *below* time-expansion, and
the residual gap is not statistically resolvable at n=1000.

## What was actually run

The real package, not a reimplementation: `pip install multiband-audio`
(**v0.1.0**, Earth Species Project, author Eklavya Sarkar — the toolkit for
Sarkar et al. 2026, *Beyond the Baseband*). Band splitting used the shipped
`mba.MultibandTransform` exactly as published:

```python
mba.MultibandTransform(sample_rate=250_000, target_sr=16_000, max_freq=32_000)
# -> 4 bands: [(0, 8k), (8k, 16k), (16k, 24k), (24k, 32k)] Hz
```

We did **not** use `mba.MultibandWrapper`. That wrapper wants a backbone
returning one `(N, D)` vector and a *learned* fusion head trained end-to-end;
our protocol is a frozen encoder plus a logistic-regression probe, and we need
per-layer hidden states. So we took the package's band waveforms and did our
own embedding + fusion. Mean-fusion is exactly their `MeanPoolFusion` ("mp");
concat-fusion is their `ConcatLinearFusion` minus the learned projection (the
probe is linear anyway). The learned fusions (`gp`, `moe`, `hyb`, `sa`) were
**not** tested — see caveats.

**Data.** 1,000 calls = 100 x 10 emitters, `groupby("Emitter").head(100)` of
`bats/subset2000.csv`. Source WAVs 250 kHz, truncated to 6 s real time
(mean call 1.99 s, so almost nothing is cut).

**Conditions** (all four on the identical 1,000 calls):

| condition | what it does | spectrum reaching the encoder |
|---|---|---|
| `baseband` | resample 250k -> 16k | 0-8 kHz |
| `timeexp4x` | relabel 250k as 62.5k, -> 16k (4x slower) | 0-31.25 kHz, folded to 0-8 kHz |
| `multiband_mean` | 4 heterodyned bands, embeddings averaged | 0-32 kHz |
| `multiband_concat` | 4 heterodyned bands, embeddings concatenated (3072-d) | 0-32 kHz |

Time-expansion and multi-band therefore see **the same spectral content**
(0-32 kHz); only the route into the 16 kHz encoder differs. That makes this a
genuinely fair head-to-head.

**Encoders.** `microsoft/wavlm-base-plus` (13 hidden states) and AVES-bio
(12). Frozen, mean-pooled over time, every input z-scored.
**Probe.** `identity_acc` from `bats_identity.py`, unchanged: stratified
5-fold, `StandardScaler` + `LogisticRegression(C=0.5, max_iter=3000)`.
Chance = 0.100.

## Results

Identity accuracy, best layer. Full per-layer curves in `results.json`.

| encoder | baseband | time-expansion 4x | multiband mean | multiband concat |
|---|---|---|---|---|
| **WavLM-base-plus** | 0.587 (L2) | **0.666** (L1) | 0.604 (L2) | 0.637 (L4) |
| **AVES-bio** | 0.608 (L9) | **0.641** (L4) | 0.561 (L2) | 0.616 (L8) |

Restricted to layers {3, 6, 9, 12} as originally scoped (AVES has no L12; L11 shown):

| encoder | layer | baseband | time-exp 4x | mb mean | mb concat |
|---|---|---|---|---|---|
| WavLM | 3 | 0.565 | **0.651** | 0.572 | 0.626 |
| WavLM | 6 | 0.496 | **0.572** | 0.499 | 0.564 |
| WavLM | 9 | 0.478 | 0.527 | 0.472 | **0.544** |
| WavLM | 12 | 0.493 | **0.526** | 0.452 | 0.521 |
| AVES | 3 | 0.582 | **0.616** | 0.549 | 0.607 |
| AVES | 6 | 0.579 | **0.639** | 0.555 | 0.604 |
| AVES | 9 | 0.608 | **0.622** | 0.545 | 0.604 |
| AVES | 11 | 0.573 | **0.622** | 0.554 | 0.592 |

Time-expansion wins at 7 of 8 of those layer/encoder cells.

### Single bands, and a built-in noise ruler

| encoder | band 0 (0-8k) | band 1 (8-16k) | band 2 (16-24k) | band 3 (24-32k) |
|---|---|---|---|---|
| WavLM | 0.600 | 0.516 | 0.485 | 0.424 |
| AVES-bio | 0.610 | 0.497 | 0.465 | 0.442 |

`band0_only` and `baseband` are the same 0-8 kHz audio down two different
resamplers (the package's librosa `kaiser_best` vs our scipy `resample_poly`).
They differ by 0.013 (WavLM) and 0.002 (AVES), both n.s. — so **~1-3 points is
this setup's noise floor**, and `band0_only` is the honest within-pipeline
reference for the fusions.

### Paired tests (`paired_tests.json`)

Same folds for every condition, so predictions pair per call. Exact McNemar,
plus a 2,000-resample paired bootstrap CI on the accuracy difference.

| comparison | WavLM | AVES-bio |
|---|---|---|
| time-exp − multiband concat | +0.029, CI [−0.002,+0.063], p=0.091 n.s. | +0.025, CI [−0.006,+0.056], p=0.13 n.s. |
| time-exp − multiband mean | +0.062, CI [+0.030,+0.096], p=3.3e-4 *** | +0.080, CI [+0.046,+0.112], p=4.5e-6 *** |
| multiband concat − band0 only | +0.037, CI [+0.005,+0.067], p=0.031 * | +0.006, CI [−0.027,+0.039], p=0.77 n.s. |
| multiband mean − band0 only | +0.004, CI [−0.028,+0.036], p=0.85 n.s. | −0.049, CI [−0.082,−0.018], p=0.0042 ** |
| multiband concat − baseband | +0.050, CI [+0.017,+0.083], p=0.003 ** | +0.008, CI [−0.023,+0.039], p=0.66 n.s. |
| baseband − band0 only | −0.013, CI [−0.041,+0.014], p=0.39 n.s. | −0.002, CI [−0.035,+0.030], p=0.95 n.s. |

## Reading of the result

1. **Multi-band never beats time-expansion.** It is behind by +0.029 (WavLM)
   and +0.025 (AVES). Both point estimates favour time-expansion and both CIs
   are near-symmetric about a small positive gap; neither reaches p<0.05. The
   defensible claim is "multi-band did not beat time-expansion here," not
   "time-expansion is significantly better."
2. **Mean-fusion is the wrong fusion for this task.** It loses decisively to
   time-expansion (p<0.001 both encoders) and, for AVES, is *worse than its own
   baseband band* (−0.049, p=0.004): averaging in three weakly-informative
   bands dilutes the one good one. Concat-fusion, which lets the probe weight
   bands, avoids this.
3. **The ultrasonic bands do carry identity, just not efficiently.** Every band
   is far above chance on its own (0.42-0.52 vs 0.10). But concat-fusion
   converts that into only +0.037 over band 0 for WavLM and +0.006 (n.s.) for
   AVES — i.e. it is largely redundant with what band 0 already has.
4. **Energy does not predict usefulness.** 76.2% of these calls' energy sits in
   band 1 (8-16 kHz), vs 10.8% in band 0 — yet band 0 alone (0.600 / 0.610)
   decisively out-probes band 1 alone (0.516 / 0.497). Whatever makes an
   individual bat identifiable to a speech encoder is not simply where the
   energy is. Measured over the first 25 calls: band 0 10.8%, band 1 76.2%,
   band 2 11.5%, band 3 0.8%, above 32 kHz 0.6% (so 4 bands cover 99.4%).

## Caveats

- **Only the two parameter-free fusions were tested.** The paper's headline is
  *adaptive* fusion — `gp`, `moe`, `hyb`, `sa` all learn per-band gates, and
  `gp` is the README's default. A learned gate would likely beat mean-fusion
  (it can down-weight bands 1-3) and could plausibly close the gap to
  time-expansion. This experiment does not test ESP's actual best method, and
  should not be cited as refuting it.
- **Shipped heterodyne is a real (single-phase) mixer, so each non-baseband
  band folds 2:1.** `HeterodyneToBaseband` multiplies by `cos(2*pi*f_center*t)`
  and takes the real result; there is no quadrature/Hilbert path. Verified with
  tones: 9 kHz and 15 kHz (both 3 kHz off the 12 kHz centre) *both* land at
  3 kHz in band 1, and a sine exactly at 12 kHz is nulled (rms 0.0006) while a
  cosine at 12 kHz passes (rms 0.500). Upper and lower sidebands are therefore
  indistinguishable within a band. This is the published behaviour, not a bug
  we introduced, but it means "multi-band" is not lossless access to the full
  spectrum.
- **Bands overlap substantially.** The band-pass is a single 2nd-order biquad
  (`Q = f_center/bandwidth`, so 1.5-3.5 here), not a brick wall. A 15 kHz tone
  appears in band 1 at rms 0.287 *and* band 2 at 0.176 — about −4 dB of
  leakage. The README's "non-overlapping frequency bands" is aspirational for
  v0.1.0.
- **Best-layer selection is not cross-validated.** Each condition is scored at
  its own argmax layer, which inflates all four conditions similarly. Ranking
  is stable across layers anyway (panel B), so this is unlikely to flip the
  conclusion.
- **Fixed `C=0.5` across dimensionalities.** Concat has 3072 features vs 768
  for the others, on 800 training samples per fold — the effective
  regularisation is not matched. Concat could be under- or over-penalised; we
  did not sweep `C`.
- **n=1000, one dataset, one task.** Half the 2,000-call set used in
  `out_bats/`, which is why every number here sits ~4 points below that run
  (WavLM baseband 0.587 here vs 0.631 there; time-expansion 0.666 vs 0.700) —
  direction and ordering replicate, magnitudes shift with training-set size.
  Identity is also not the task the paper targets (species/call-type
  classification); a task whose discriminative content genuinely lives above
  8 kHz would be a fairer test of multi-band.
- **No session/recording holdout.** Same limitation as `out_bats/`: identity
  labels may be partly carried by per-session recording conditions.

## Files

| file | contents |
|---|---|
| `results.json` | per-layer accuracy, best layer, mean, and the {3,6,9,12} subset, for 8 conditions x 2 encoders |
| `paired_tests.json` | McNemar + bootstrap CIs for the 6 comparisons above |
| `multiband.png` | A: best accuracy per condition; B: per-layer curves; C: single-band accuracy |
| `meta.json` | the 1,000 file names, emitter labels, band edges, offsets |
| `emb_<enc>_<cond>.npy` | cached embeddings, `(layers, 1000, 768)` float16 |
| `wav_bands.npy`, `wav_baseband.npy`, `wav_timeexp4x.npy` | cached waveforms, float16, unit-std |

Reproduce: `multiband_run.py` (wavlm \| aves-bio), then `multiband_stats.py`,
then `multiband_fig.py`. Band-splitting characterisation: `mb_sanity.py`,
`mb_sanity2.py`.
