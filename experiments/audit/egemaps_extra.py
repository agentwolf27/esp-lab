"""
eGeMAPSv02 functionals (openSMILE, 88-d) for DOGS and PIGS.

Cats already have theirs (ctx/egemaps.py -> out_enc/results.json["egemaps"]).
Audio preprocessing is identical to the WavLM path for each dataset
(mono, 16 kHz, 6 s truncation) so the handcrafted and neural rows see the
same signal.

Order is locked to the corresponding meta file so labels line up:
    dogs -> out_audit/meta_dogs.json      (693 barks)
    pigs -> out_pigs/meta.json            (5,031 calls)

Writes out_audit/egemaps_dogs.npy, out_audit/egemaps_pigs.npy (float32).

    python egemaps_extra.py dogs|pigs
"""
from __future__ import annotations

import json, os, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
CTX = os.path.dirname(HERE)
SR = 16_000
MAX_S = 6.0


def read16k(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    x = x[: int(MAX_S * SR)]
    if len(x) < 1600:
        x = np.pad(x, (0, 1600 - len(x)))
    return x


def run(which):
    import opensmile
    sm = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                         feature_level=opensmile.FeatureLevel.Functionals)
    if which == "cats":
        # recomputed here (ctx/egemaps.py did not save the feature matrix, and
        # used C=0.5); order locked to out_enc/meta.json, i.e. the same clip
        # list and order the cached encoder embeddings use.
        import glob
        from scipy.signal import resample_poly  # noqa: F401  (used in read16k)
        import soundfile as sf
        meta = json.load(open(os.path.join(CTX, "out_enc", "meta.json")))["meta"]
        files = sorted(glob.glob(os.path.join(CTX, "catmeows", "**", "*.wav"), recursive=True))
        paths = []
        for f in files:
            x, sr = sf.read(f, dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            if sr != SR:
                g = np.gcd(int(sr), SR)
                x = resample_poly(x, SR // g, sr // g).astype(np.float32)
            if len(x) < 800:      # same drop rule as encoders.load_clips
                continue
            paths.append(f)
        assert len(paths) == len(meta), (len(paths), len(meta))
        for p, m in zip(paths, meta):
            assert os.path.basename(p)[:-4].split("_")[1] == m["cat"]
    elif which == "dogs":
        meta = json.load(open(os.path.join(HERE, "meta_dogs.json")))
        paths = [os.path.join(CTX, "dogs", "audio", m["file"]) for m in meta]
    else:
        meta = json.load(open(os.path.join(CTX, "out_pigs", "meta.json")))
        adir = os.path.join(CTX, "pigs", "Soundwel Dataset - Audio and Spectrograms")
        paths = [os.path.join(adir, m["file"]) for m in meta]

    X, t0 = [], time.time()
    for i, p in enumerate(paths):
        x = read16k(p)
        X.append(sm.process_signal(x, SR).values[0])
        if (i + 1) % 250 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(paths)}  {el:.0f}s  eta {el/(i+1)*(len(paths)-i-1):.0f}s", flush=True)
    X = np.nan_to_num(np.array(X, dtype=np.float32))
    np.save(os.path.join(HERE, f"egemaps_{which}.npy"), X)
    print(f"wrote egemaps_{which}.npy {X.shape}", flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
