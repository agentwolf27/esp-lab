# Data (not committed)

All audio lives outside the repo. Every dataset below is public; each carries its own license,
which you must respect. Scripts under `experiments/context_probe/` expect these layouts.

| dataset | what | get it | license |
|---|---|---|---|
| **CatMeows** | 440 meows, 21 cats, 3 contexts (brushing / waiting-for-food / isolation) | Zenodo `4008297` → `dataset.zip` → `experiments/context_probe/catmeows/` | CC-BY-4.0 |
| **Dog barks** (Molnár et al. 2008) | 693 barks, 10 dogs, contexts contact / play / aggression | `https://storage.googleapis.com/ml-bioacoustics-datasets/dog_barks.zip` (via Earth Species Project BEANS) → `experiments/context_probe/dogs/` | see BEANS |
| **Soundwel** (Briefer et al. 2022) | 6,887 pig calls, 17 contexts, valence labels, 6 labs | Zenodo `8252482` → `experiments/context_probe/pigs/` (+ `SoundwelDatasetKey.xlsx` as `key.xlsx`) | CC-BY-4.0 |
| **Egyptian fruit bats** (Prat et al. 2017) | 10 emitters × 1,000 calls at 250 kHz | `https://storage.googleapis.com/ml-bioacoustics-datasets/egyptian_fruit_bats.zip` (via BEANS; 5.2 GB) → `experiments/context_probe/bats/` | see Prat et al. 2017 / figshare `c.3666502` |
| **BirdSet** (for `conformal_pam/`) | soundscape eval sets POW / HSN | HF `mteb/BirdSet`, split `test_5s` — fetched inside the Kaggle script | see BirdSet |

Model weights are also not committed. `AVES-bio` (Earth Species Project) is downloaded by
`encoders.py` from ESP's public GCS bucket; its weights are **CC-BY-NC-SA-4.0** — research use only.
Speech encoders (WavLM, HuBERT, wav2vec2) come from the Hugging Face Hub on first run.
