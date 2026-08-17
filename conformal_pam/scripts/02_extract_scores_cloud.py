"""
STAGE 1 (cloud, run once): audio -> score matrix.

Paste this into a Kaggle or Colab notebook. It downloads two BirdSet soundscape
sites, runs the frozen Perch 2.0 classifier over every 5-second window, and
saves a compact .npz you bring back to the laptop. Nothing after this needs a
GPU ever again.

    input   ~0.7 GB of audio   (two sites)
    output  ~60-120 MB .npz    (scores + labels + 1536-d embeddings + timestamps)

Why these choices
-----------------
Perch 2.0 is the only model in this space with **Apache-2.0 code AND weights**,
so results built on it stay usable commercially and are redistributable. Every
alternative is encumbered: BirdNET weights are CC-BY-NC-SA, AudioProtoPNet is
CC-BY-NC, and Bird-MAE and ConvNeXT-BirdSet ship with no license at all.

BirdSet's test sets are per-site soundscapes, which hands us the
calibrate-on-site-A / deploy-on-site-B axis directly -- that is the whole
experiment.

We use the `mteb/BirdSet` parquet mirror rather than the official repo, because
the official one is a script dataset that the HF datasets-server now refuses.

Site sizes (test_5s split, verified):
    POW  0.21 GB    4,560 windows   Powdermill, Pennsylvania
    HSN  0.49 GB   12,000 windows   High Sierra Nevada, California
    UHH  1.54 GB   36,637 windows   Hawaii
    SSW  8.54 GB  205,200 windows   Sapsucker Woods, New York
Start with POW + HSN. That is a real two-site experiment for 0.7 GB.
"""

# ---------------------------------------------------------------- setup
# !pip install -q datasets soundfile huggingface_hub
# Kaggle/Colab already have tensorflow and numpy.

import io
import json

import numpy as np
import soundfile as sf
from datasets import load_dataset

SITES = ["POW", "HSN"]          # add "UHH" once the two-site version works
SPLIT = "test_5s"               # pre-windowed 5s clips, already the model's frame
TARGET_SR = 32_000              # Perch 2.0 operates at 32 kHz
WINDOW = 5 * TARGET_SR          # 160,000 samples
BATCH = 64
OUT = "birdset_perch_scores.npz"


# ---------------------------------------------------------------- model
def load_perch():
    """Load Perch 2.0. Two routes; the HF mirror needs no Kaggle credentials.

    NOTE: this loader is the one part of the pipeline not executed locally
    (there is no GPU or TensorFlow on the laptop). It prints the discovered
    signature so a mismatch is obvious immediately rather than 40 minutes in.
    """
    import tensorflow as tf

    try:                                            # route 1: official wrapper
        from perch_hoplite.zoo import model_configs
        m = model_configs.load_model_by_name("perch_v2")
        print("loaded via perch_hoplite")
        return ("hoplite", m)
    except Exception as e:                          # route 2: Apache-2.0 HF mirror
        print(f"perch_hoplite unavailable ({type(e).__name__}), using HF mirror")
        from huggingface_hub import snapshot_download

        path = snapshot_download("cgeorgiaw/Perch")
        m = tf.saved_model.load(path)
        sig = m.signatures["serving_default"]
        print("saved_model inputs :", {k: v.shape for k, v in sig.structured_input_signature[1].items()})
        print("saved_model outputs:", {k: v.shape for k, v in sig.structured_outputs.items()})
        return ("savedmodel", m)


def score_batch(kind, model, waveforms: np.ndarray):
    """(B, 160000) float32 -> (logits (B, n_classes), embeddings (B, 1536)).

    We dump BOTH. The logits feed the conformal work; the 1536-d embeddings
    unlock everything in IDEAS.md that comes later (sparse autoencoders,
    temporal-context models, cascades, open-set) without ever touching audio
    again. One extra array, ~6 kB per window in float16.
    """
    import tensorflow as tf

    if kind == "hoplite":
        out = model.batch_embed(waveforms)
        # perch_hoplite applies logits = raw * 0.97 - 10.0 on top of the
        # SavedModel output. It is a monotone affine map, so conformal
        # quantiles and coverage are unaffected -- but it WILL change any
        # temperature/Platt fit, so record which convention produced a file.
        logits = np.asarray(out.logits["label"], dtype=np.float32)
        emb = np.asarray(out.embeddings, dtype=np.float32)
        return logits, emb.reshape(emb.shape[0], -1)

    sig = model.signatures["serving_default"]
    res = sig(tf.constant(waveforms, dtype=tf.float32))
    lkey = next(k for k in res if "logit" in k.lower() or "label" in k.lower())
    ekey = next((k for k in res if "embed" in k.lower()), None)
    logits = np.asarray(res[lkey], dtype=np.float32)
    emb = (np.asarray(res[ekey], dtype=np.float32).reshape(logits.shape[0], -1)
           if ekey else np.zeros((logits.shape[0], 0), np.float32))
    return logits, emb


# ---------------------------------------------------------------- audio
def decode(example) -> np.ndarray:
    """Decode one ogg blob to exactly WINDOW samples of 32 kHz mono."""
    audio = example["audio"]
    wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")

    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:                              # BirdSet is already 32 kHz
        idx = np.linspace(0, len(wav) - 1, int(len(wav) * TARGET_SR / sr))
        wav = np.interp(idx, np.arange(len(wav)), wav).astype(np.float32)

    # Fixed length, always. Never pad to "longest in batch" -- that makes a
    # clip's features depend on what else is in its batch, which silently
    # breaks the exchangeability that the whole conformal guarantee rests on.
    if len(wav) < WINDOW:
        wav = np.pad(wav, (0, WINDOW - len(wav)))
    return wav[:WINDOW]


# ---------------------------------------------------------------- main
def main() -> None:
    kind, model = load_perch()

    all_logits, all_emb = [], []
    all_labels, all_rec, all_site, all_path, all_lat, all_lon = [], [], [], [], [], []

    for site in SITES:
        print(f"\n=== {site} ===")
        ds = load_dataset("mteb/BirdSet", site, split=SPLIT)
        print(f"  {len(ds):,} windows")

        buf, meta = [], []
        for i, ex in enumerate(ds):
            buf.append(decode(ex))
            meta.append((ex.get("ebird_code_multilabel") or [],
                         ex["audio"]["path"],
                         ex.get("lat"), ex.get("long")))

            if len(buf) == BATCH or i == len(ds) - 1:
                lg, em = score_batch(kind, model, np.stack(buf))
                all_logits.append(lg)
                all_emb.append(em.astype(np.float16))   # halves the file, loses nothing we need
                for lab, path, lat, lon in meta:
                    all_labels.append(lab)
                    # Recording identity, NOT window identity. Splitting by
                    # recording is what stops windows from the same file
                    # landing in both calibration and test -- the leak that
                    # would make every number look wonderful and mean nothing.
                    all_rec.append("_".join(path.split("_")[:4]))
                    all_site.append(site)
                    # Full path keeps the timestamp + window offset, e.g.
                    # 'UHH_001_S01_20161121_150000_000_005.ogg' -> date, time,
                    # start/end seconds. That is what makes weather joins
                    # (IDEAS 2.2) and temporal-context models (IDEAS 2.4)
                    # possible later without re-running anything.
                    all_path.append(path)
                    all_lat.append(np.nan if lat is None else float(lat))
                    all_lon.append(np.nan if lon is None else float(lon))
                buf, meta = [], []
                if (i + 1) % (BATCH * 20) == 0:
                    print(f"  {i + 1:,}/{len(ds):,}")

    logits = np.concatenate(all_logits).astype(np.float32)
    embeddings = np.concatenate(all_emb)
    print(f"\nlogits {logits.shape}   embeddings {embeddings.shape}")

    # Multi-label targets, densified only over species that actually occur
    # here. Perch has 14,795 classes; two sites use a few hundred. Keeping all
    # of them would make the file 30x bigger for no information.
    present = sorted({c for lab in all_labels for c in lab})
    col = {c: j for j, c in enumerate(present)}
    Y = np.zeros((len(all_labels), len(present)), dtype=bool)
    for i, lab in enumerate(all_labels):
        for c in lab:
            Y[i, col[c]] = True

    S = logits[:, np.array(present, dtype=int)] if present else logits[:, :0]

    np.savez_compressed(
        OUT,
        logits=S,
        labels=Y,
        embeddings=embeddings,                 # (N, 1536) float16 -- IDEAS 2.1/2.3/2.4
        recording=np.array(all_rec),
        site=np.array(all_site),
        path=np.array(all_path),               # timestamps live in here -- IDEAS 2.2/2.4/2.8
        lat=np.array(all_lat, dtype=np.float32),
        lon=np.array(all_lon, dtype=np.float32),
        class_index=np.array(present),
        meta=json.dumps({
            "model": "perch_v2",
            "loader": kind,
            "affine_applied": kind == "hoplite",   # raw*0.97 - 10.0
            "sites": SITES, "split": SPLIT, "sr": TARGET_SR,
            "window_s": 5, "n_full_classes": int(logits.shape[1]),
            "embedding_dim": int(embeddings.shape[1]),
        }),
    )
    print(f"\nwrote {OUT}: {S.shape[0]:,} windows x {S.shape[1]} species "
          f"+ {embeddings.shape[1]}-d embeddings")
    print(f"positives: {int(Y.sum()):,}   recordings: {len(set(all_rec))}")
    print("\nDownload that file and run 03_real_analysis.py on the laptop.")


if __name__ == "__main__":
    main()
