# Multi-dataset audit — identity leakage and evaluation inflation

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

| Dataset | Features | Held-out group | Random | Inflation | Group identity | Chance (task / identity) |
|---|---|---|---|---|---|---|
| CatMeows | WavLM-base-plus (L3) | **0.559** | 0.725 | +0.167 | 0.793 | 0.333 / 0.048 |
| CatMeows | HuBERT-base (L12) | **0.571** | 0.671 | +0.099 | 0.700 | 0.333 / 0.048 |
| CatMeows | wav2vec2-base (L2) | **0.554** | 0.737 | +0.183 | 0.755 | 0.333 / 0.048 |
| CatMeows | AVES-bio (L5) | **0.540** | 0.736 | +0.196 | 0.793 | 0.333 / 0.048 |
| CatMeows | eGeMAPS-88 | **0.492** | 0.628 | +0.136 | 0.727 | 0.333 / 0.048 |
| CatMeows | MFCC-85 | **0.399** | 0.602 | +0.203 | 0.768 | 0.333 / 0.048 |
| Dog barks | WavLM-base-plus (L6) | **0.728** | 0.878 | +0.150 | 0.817 | 0.333 / 0.100 |
| Dog barks | eGeMAPS-88 | **0.619** | 0.733 | +0.114 | 0.792 | 0.333 / 0.100 |
| Dog barks | MFCC-85 | **0.551** | 0.793 | +0.242 | 0.892 | 0.333 / 0.100 |
| Dog barks | log-duration only * | **0.341** | 0.360 | +0.018 | 0.222 | 0.333 / 0.100 |
| Soundwel pigs | WavLM-base-plus (L0) | **0.660** | 0.883 | +0.223 | 0.938 | 0.500 / 0.167 |
| Soundwel pigs | eGeMAPS-88 | **0.669** | 0.823 | +0.153 | 0.865 | 0.500 / 0.167 |
| Soundwel pigs | paper's 18 features | **0.386** | 0.575 | +0.189 | 0.513 | 0.500 / 0.167 |
| BEANS bats (exploratory) | WavLM-base-plus (L11) | **0.301** | 0.303 | +0.003 | 0.394 | 0.250 / 0.100 |

`*` log-duration only, included as a floor: dog contexts are **not** simply
different recording lengths.

Mean over all layers, for the multi-layer encoders:

| Dataset | Encoder | Layers | Held-out (mean) | Random (mean) | Inflation (mean) | Identity (mean) |
|---|---|---|---|---|---|---|
| CatMeows | WavLM-base-plus | 13 | 0.525 | 0.667 | +0.142 | 0.720 |
| CatMeows | HuBERT-base | 13 | 0.522 | 0.694 | +0.172 | 0.741 |
| CatMeows | wav2vec2-base | 13 | 0.519 | 0.669 | +0.150 | 0.689 |
| CatMeows | AVES-bio | 12 | 0.520 | 0.731 | +0.211 | 0.809 |
| Dog barks | WavLM-base-plus | 13 | 0.698 | 0.861 | +0.164 | 0.819 |
| Soundwel pigs | WavLM-base-plus | 13 | 0.604 | 0.813 | +0.208 | 0.824 |
| BEANS bats (exploratory) | WavLM-base-plus | 13 | 0.273 | 0.289 | +0.016 | 0.421 |

## The inflation law

Pooling every (dataset, feature set, layer) point, excluding the exploratory
bats and the duration floor:

* **pooled** n=83, Pearson r=0.611 (p=8.73e-10),
  Spearman rho=0.610 (p=9.36e-10)
* **within-dataset partial** (each dataset centred on its own mean)
  r=0.646, df=79, p=7.10e-11
* **per dataset**: cats r=0.712 (n=53, p=2.27e-09); dogs r=0.587 (n=15, p=0.022); pigs r=0.595 (n=15, p=0.019)
* **cluster level**, one point per (dataset × feature set):
  n=12, r=0.221 (p=0.489),
  rho=0.441 (p=0.152) — **not significant**
* **neural points only** (768-d, so the effect is not "handcrafted vs deep"):
  n=77, r=0.694 (p=2.49e-12);
  within-dataset r=0.736 (p=5.64e-14)
* **cats, one point per feature set** (the n=6 successor to the earlier n=5 test that gave
  r=0.81, p=0.094): r=0.866, p=0.026
* **within one encoder** — centred inside each (dataset × feature set) cluster, so the only
  variation left is layer-to-layer inside a single frozen model:
  n=77, 6 clusters,
  df=70, r=0.636,
  p=1.99e-09. Per encoder: cats/aves-bio r=+0.36 (p=0.248); cats/hubert r=+0.03 (p=0.921); cats/wav2vec2 r=+0.90 (p=2.4e-05); cats/wavlm r=+0.79 (p=0.001); dogs/wavlm r=+0.38 (p=0.199); pigs/wavlm r=+0.86 (p=1.6e-04) — all six positive
  (sign test p=0.031), three individually significant.
* **leverage**: dropping the most extreme point (pigs/acoustic18) *raises*
  the pooled r to 0.670 (p=5.65e-12);
  jackknifing out each feature set in turn keeps r in
  [0.480, 0.689].

In `inflation_law.png` colour is the dataset and marker shape is the feature set
(circle WavLM, square HuBERT, triangle wav2vec2, diamond AVES-bio, plus eGeMAPS,
cross MFCC-85, star the pigs' published 18 features).

Decomposition: identity tracks the **leaky** number
(r=0.754, p=1.87e-16) more
strongly than the honest one
(r=0.509, p=8.84e-07),
which is the direction the leakage account predicts.

Including the exploratory bats lifts the pooled correlation to
r=0.874 (n=96, p=2.72e-31), but that is a
leverage artefact: within bats the correlation is *negative*
(r=-0.261, p=0.390) and no bat
context label is decodable at all, so their inflation is a degenerate zero.

## How much labelled target-group data repairs it

For each group a fixed stratified **test half** is set aside once; k labelled clips are drawn
from the other half and added to the training set, so k=0 and k>0 are scored on exactly the
same clips. k is capped at the pool size — the median cat only has ~19 clips in total.

| Dataset · features | k=0 | k=10 | k=25 | k=50 | k=100 | random-split ceiling | effective k |
|---|---|---|---|---|---|---|---|
| CatMeows · WavLM L3 | 0.576 | 0.677 | 0.727 | 0.726 | 0.727 | 0.729 | 0/8/11/11/11 |
| CatMeows · eGeMAPS | 0.525 | 0.581 | 0.587 | 0.592 | 0.592 | 0.628 | 0/8/11/11/11 |
| Soundwel pigs · WavLM L0 | 0.654 | 0.679 | 0.706 | 0.730 | 0.769 | 0.884 | 0/10/25/50/100 |
| Soundwel pigs · eGeMAPS | 0.677 | 0.682 | 0.686 | 0.693 | 0.704 | 0.823 | 0/10/25/50/100 |

Accuracies are the mean over 10 (cats) / 5 (pigs) resamples of which clips are added; sd is in `recovery.json` and is ≤0.016 everywhere.

Cats: ~11 clips from the target cat is enough to reach the random-split ceiling
(0.576 → 0.727 vs a
0.729 ceiling), i.e. the whole "shift" is target-individual
idiosyncrasy that a handful of labels absorbs. Pigs: 100 clips from the target lab closes only
50%
of the gap and is still climbing — lab shift is deeper than individual shift.
eGeMAPS recovers far less in both, i.e. the encoder's advantage is partly that it is
*adaptable*, not that it is *transferable*.

## Verdict

Supported WITHIN a dataset, not established ACROSS datasets. Identity decodability predicts inflation among feature sets and layers of the same dataset (within-dataset partial r=0.646, df=79, p=7.1e-11; within-encoder, layer-only r=0.636, df=70, p=2.0e-9; all six encoders positive), but the conservative cluster-level test over the 12 independent (dataset x feature set) units is null (r=0.221, p=0.489), so 'more identity leakage => more inflation' is a within-corpus selection rule, not a cross-corpus law.

## Files

`results.json` (everything), `audit_table.png`, `inflation_law.png`, `recovery.png`,
`points.json` (per-layer points), `law.json`, `recovery.json`.
Scripts: `embed_dogs.py`, `egemaps_extra.py`, `audit.py`, `report.py`.
Caches: `emb_dogs_wavlm.npy` (13×693×768), `egemaps_{cats,dogs,pigs}.npy`, `mfcc_dogs.npy`.

## Caveats

* Layer points from one encoder are not independent; the pooled p-values are anti-conservative. The cluster-level test (one point per dataset x feature set) is the conservative version and it is not significant (n=12, r=0.221, p=0.489).
* Identity also correlates with the honest held-out score (r=0.509, p=8.8e-07), not only with the leaky one (r=0.754, p=1.9e-16). Part of the relation is therefore 'better features decode everything better', not pure leakage; the leakage account survives only because the leaky number moves more.
* Bats are exploratory only: no context label is decodable at all (held-out 0.25-0.30 against a 0.25 chance), so their near-zero inflation is a degenerate zero, not support for the law. Including bats raises the pooled r to 0.874 purely as a leverage effect; the primary fit excludes them.
* Dog 'context' is confounded with recording session, and dogs contribute only 10 groups; the leave-one-dog-out number is not a leave-one-session-out number.
* Cat and pig headline rows are re-used from the earlier runs (out_enc, out_pigs). Pig WavLM layer 0 was recomputed here as a check and matched to <0.002.
* Recovery uses a fixed stratified test half per group so k=0 and k>0 are scored on exactly the same clips; k is capped at the adaptation pool size.
