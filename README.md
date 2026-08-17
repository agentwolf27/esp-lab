# esp-lab

Laptop-scale experiments on **frozen audio encoders and animal vocalisations** — what they actually
encode, how evaluation leaks, and whether anything about *affect* transfers across species.

Everything here ran on an Apple M2 with 8 GB RAM, CPU only, $0 of compute. Started 15 Aug 2026 after
reading everything the [Earth Species Project](https://earthspecies.org) has open-sourced.

**Read the results:** [`FINDINGS.md`](FINDINGS.md) · or the illustrated version:
[artifacts/opposite-signs.html](artifacts/opposite-signs.html) (open locally).

## What's in here

| | |
|---|---|
| `FINDINGS.md` | the full log of results, controls, an adversarial review, and what got downgraded or killed |
| `experiments/context_probe/` | every script, in the order they were run (see below) |
| `conformal_pam/` | conformal risk control for passive-acoustic detections (site-shift chapter; Kaggle step pending) |
| `results/` | JSON outputs of every run — the numbers in FINDINGS trace to these |
| `figures/` | the plots |
| `artifacts/` | three self-contained HTML pages: ESP capability map, first probe, findings |
| `data/README.md` | how to fetch each dataset (nothing large is committed) |

## The findings, in one paragraph

Across **cats** (440 meows, 21 individuals), **dogs** (693 barks, 10 individuals), **pigs** (5,031 calls,
6 labs) and **wild bats** (2,000 calls, 10 emitters), frozen encoders — WavLM, HuBERT, wav2vec2,
ESP's AVES — carry **individual identity and recording site far more strongly than behavioural
context**. Identity is decodable at 0.63–0.79 (chance 0.05–0.10) and lab at 0.82 (chance 0.17).
Evaluating with random splits therefore inflates "context accuracy" by **+0.10 to +0.21**; on the pig
data, the published acoustic features fall **below chance** once a lab is held out. That is the strong
result. A second, smaller one: an affect axis learned on dog barks predicts cat contexts (0.58,
animal-level p=0.0045, placebo-clean, survives balanced classes and strict scaling); about half of it is
call duration; it flows *into* cats but not out; and it is only partly stable across encoders.

## Running it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# fetch datasets per data/README.md, then, from experiments/context_probe/:
python context_probe.py catmeows out        # baseline, seconds
python encoders.py                          # 5 frozen encoders, leave-one-cat-out
python cross_species.py                     # dog<->cat affect transfer
python validate_transfer.py                 # permutation nulls, nested selection, controls
python killtest.py                          # duration, animal-level permutation, asymmetry
python placebo.py                           # specificity + residual + AUC
python pigs_run.py                          # third species, 3x3 matrix
python bats_identity.py                     # wild-species identity + bandwidth
```

Each script is documented at the top with what it tests and why. Splits are always by
**individual or lab, never random** — that rule is the whole point.

## Status

Work in progress (Aug 2026). Two directions: an evaluation-leakage audit across species (the strong
result), and a cautious cross-species affect-transfer study (the modest one). Issues and corrections
welcome — the adversarial-review section of `FINDINGS.md` shows what already got corrected.

## Acknowledgements

Datasets: CatMeows (Ludovico et al. 2021), dog barks (Molnár et al. 2008), Soundwel (Briefer et al.
2022), Egyptian fruit bats (Prat et al. 2017); the last two reached me via the Earth Species Project's
[BEANS](https://github.com/earthspecies/beans) benchmark. Encoders: Microsoft WavLM, Meta HuBERT /
wav2vec2, ESP AVES. Code is MIT; data and weights carry their own licenses (see `data/README.md`).
