"""
Dogs: WavLM-base-plus mean-pooled per-layer embeddings for ALL 693 barks.

The cached out_cross/emb_wavlm.npy only covers the 2-class subset (play /
aggression) used for the cross-species transfer test.  The audit needs the
full 3-class context task (contact / play / aggression), so we re-embed.

Audio handling matches cross_species.py exactly:
    stereo -> mono mean, resample to 16 kHz, truncate at MAX_S = 6.0 s,
    per-clip mean/std normalisation, batch size 1 (padding changes
    HuBERT-family embeddings).

Writes out_audit/emb_dogs_wavlm.npy  (13, 693, 768) float16
       out_audit/meta_dogs.json
"""
from __future__ import annotations

import json, os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CTX = os.path.dirname(HERE)
SR = 16_000
MAX_S = 6.0


def load_dogs():
    import soundfile as sf
    from scipy.signal import resample_poly

    ann = pd.read_csv(os.path.join(CTX, "dogs", "annotations.csv"))
    wavs, meta = [], []
    for _, r in ann.iterrows():
        p = os.path.join(CTX, "dogs", "audio", r["filename"])
        try:
            x, sr = sf.read(p, dtype="float32")
        except Exception as e:                                   # noqa: BLE001
            print("skip", r["filename"], e, flush=True)
            continue
        if x.ndim > 1:
            x = x.mean(1)
        if sr != SR:
            g = np.gcd(int(sr), SR)
            x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        dur_full = len(x) / SR
        x = x[: int(MAX_S * SR)]
        if len(x) < 1600:
            x = np.pad(x, (0, 1600 - len(x)))
        wavs.append(x)
        meta.append({"file": r["filename"], "dog": r["name"], "ctx": r["context"],
                     "sex": r["sex"], "breed": r["breed"],
                     "dur_full": float(dur_full), "dur_used": float(len(x) / SR)})
    return wavs, meta


def embed(wavs, model_id="microsoft/wavlm-base-plus"):
    import torch
    torch.set_num_threads(2)
    from transformers import AutoModel

    m = AutoModel.from_pretrained(model_id, output_hidden_states=True).eval()
    out, t0 = [], time.time()
    with torch.no_grad():
        for i, x in enumerate(wavs):
            t = torch.from_numpy(x).float().unsqueeze(0)
            t = (t - t.mean()) / (t.std() + 1e-7)
            hs = m(t).hidden_states
            out.append(np.stack([h[0].mean(0).numpy() for h in hs]))
            if (i + 1) % 50 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(wavs)}  {el:.0f}s  eta {el/(i+1)*(len(wavs)-i-1):.0f}s", flush=True)
    del m
    return np.stack(out, axis=1)                                  # (L+1, N, D)


if __name__ == "__main__":
    wavs, meta = load_dogs()
    print(f"{len(wavs)} dog barks loaded; "
          f"mean used dur {np.mean([m['dur_used'] for m in meta]):.2f}s "
          f"(full {np.mean([m['dur_full'] for m in meta]):.2f}s)", flush=True)
    json.dump(meta, open(os.path.join(HERE, "meta_dogs.json"), "w"))
    E = embed(wavs)
    np.save(os.path.join(HERE, "emb_dogs_wavlm.npy"), E.astype(np.float16))
    print("wrote emb_dogs_wavlm.npy", E.shape, flush=True)
