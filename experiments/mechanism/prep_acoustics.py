"""Cache 4 interpretable acoustic features for cat/dog/pig, matching the
exact preprocessing that produced the cached embeddings (16 kHz mono, first 6 s).

Features (same definition for all three species):
    [log_duration, log_rms_energy, spectral_centroid, zero_crossing_rate]

Cat/dog: reuses validate_transfer.acoustic_features on cross_species.load_all().
Pig    : streams the same 5031 files listed in out_pigs/meta.json, one at a time,
         so peak memory stays a few MB.

Writes out_sae/acoustics.npz  (never touches any existing out_*/ file).
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, D)
OUT = os.path.join(D, "out_sae")
SR = 16_000
MAX_S = 6.0


def feats_from_wav(x):
    from scipy.signal import stft
    f, _, Z = stft(x, fs=SR, nperseg=400, noverlap=240, nfft=512)
    P = np.abs(Z) ** 2
    cent = (P * f[:, None]).sum(0) / (P.sum(0) + 1e-12)
    zcr = np.mean(np.abs(np.diff(np.sign(x))) > 0)
    return [np.log(len(x) / SR + 1e-3), np.log(np.sqrt((x ** 2).mean()) + 1e-8),
            float(cent.mean()), float(zcr)]


def main():
    t0 = time.time()
    # ---- cat / dog
    import cross_species as cs
    from validate_transfer import acoustic_features
    rows = cs.load_all()
    meta_cd = json.load(open(os.path.join(D, "out_cross", "meta.json")))
    assert len(rows) == len(meta_cd), (len(rows), len(meta_cd))
    for r, m in zip(rows, meta_cd):          # row order must match the cached embeddings
        assert os.path.basename(r["path"]) == os.path.basename(m["path"]), "row order mismatch"
    A_cd = acoustic_features(rows).astype(np.float32)
    print(f"cat/dog acoustics {A_cd.shape}  ({time.time()-t0:.0f}s)", flush=True)
    del rows

    # ---- pigs (streamed)
    import soundfile as sf
    from scipy.signal import resample_poly
    pm = json.load(open(os.path.join(D, "out_pigs", "meta.json")))
    adir = os.path.join(D, "pigs", "Soundwel Dataset - Audio and Spectrograms")
    A_pig = np.zeros((len(pm), 4), dtype=np.float32)
    for i, m in enumerate(pm):
        x, sr = sf.read(os.path.join(adir, m["file"]), dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        if sr != SR:
            g = np.gcd(int(sr), SR)
            x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        x = x[: int(MAX_S * SR)]
        if len(x) < 1600:
            x = np.pad(x, (0, 1600 - len(x)))
        A_pig[i] = feats_from_wav(x)
        if (i + 1) % 1000 == 0:
            print(f"  pig {i+1}/{len(pm)}  ({time.time()-t0:.0f}s)", flush=True)

    np.savez(os.path.join(OUT, "acoustics.npz"), catdog=A_cd, pig=A_pig)
    # sanity: computed pig duration vs the key's own Dur column
    key_dur = np.array([m["acoustic"][5] for m in pm], dtype=np.float32)
    ok = np.isfinite(key_dur) & (key_dur > 0)
    r = np.corrcoef(A_pig[ok, 0], np.log(key_dur[ok]))[0, 1]
    print(f"pig log-dur vs key Dur pearson r = {r:.4f}  (truncation at {MAX_S}s)")
    print(f"wrote {OUT}/acoustics.npz  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
