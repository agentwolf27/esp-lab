# Data (not committed)

All audio lives outside the repo. Every dataset below is public; each carries its own license,
which you must respect. Scripts under `experiments/context_probe/` expect these layouts.

| dataset | what | get it | license |
|---|---|---|---|
| **CatMeows** | 440 meows, 21 cats, 3 contexts (brushing / waiting-for-food / isolation) | Zenodo `4008297` → `dataset.zip` → `experiments/context_probe/catmeows/` | CC-BY-4.0 |
| **Dog barks** (Molnár et al. 2008) | 693 barks, 10 dogs, contexts contact / play / aggression | `https://storage.googleapis.com/ml-bioacoustics-datasets/dog_barks.zip` (via Earth Species Project BEANS) → `experiments/context_probe/dogs/` | see BEANS |
| **Soundwel** (Briefer et al. 2022) | 6,887 pig calls, 17 contexts, valence labels, 6 labs | Zenodo `8252482` → `experiments/context_probe/pigs/` (+ `SoundwelDatasetKey.xlsx` as `key.xlsx`) | CC-BY-4.0 |
| **Egyptian fruit bats** (Prat et al. 2017) | 10 emitters × 1,000 calls at 250 kHz | `https://storage.googleapis.com/ml-bioacoustics-datasets/egyptian_fruit_bats.zip` (via BEANS; 5.2 GB) → `experiments/context_probe/bats/` | see Prat et al. 2017 / figshare `c.3666502` |
| **Egyptian fruit bats — full corpus metadata** | 91,080 annotations, 82 identified emitters, 345 days, 21 treatments, 12 mic channels | two small CSVs, **no audio**: `https://ndownloader.figshare.com/files/8900695` (FileInfo.csv, 31.6 MB) and `https://ndownloader.figshare.com/files/7379008` (Annotations.csv, 3.3 MB) → anywhere; run `experiments/bats_session/design.py --dir .` | as above |
| **Egyptian fruit bats — full corpus audio** | 293,238 WAV at 250 kHz, 97.8 GB in 31 archives | figshare collection `c.3666502`, one item per folder (~3.2 GB each). The annotated calls touch 30 of the 31 archives, so there is no partial download that saves much — see below | as above |
| **BirdSet** (for `conformal_pam/`) | soundscape eval sets POW / HSN | HF `mteb/BirdSet`, split `test_5s` — fetched inside the Kaggle script | see BirdSet |

Model weights are also not committed. `AVES-bio` (Earth Species Project) is downloaded by
`encoders.py` from ESP's public GCS bucket; its weights are **CC-BY-NC-SA-4.0** — research use only.
Speech encoders (WavLM, HuBERT, wav2vec2) come from the Hugging Face Hub on first run.

## Note on the bat corpus

The BEANS convenience download (10 emitters × 1,000 calls) omits `FileInfo.csv`, which is the only
place the **recording time, recording channel and treatment period** live. Without it there is no
session to hold out, which is why `experiments/context_probe/bats_identity.py` records that as a
limitation. The file is public and small; `experiments/bats_session/design.py` joins it to the
annotations and reports whether a session-held-out individual-ID audit is possible.

Measured 18 Aug 2026: **82 of 82 identified emitters recur across ≥2 distinct days** (median 62),
so a day-held-out split scores every individual. Cramér's V between the emitter label and the
recording channel is **0.589** — almost exactly the 0.59 we measure between pig valence and
recording lab.
