"""
Experiment 1b: do real frozen encoders beat the 0.39 hand-crafted floor?

Encoders compared on CatMeows (440 meows, 21 cats, 3 contexts):
  mfcc          85-d hand-crafted baseline           (the floor: 0.391)
  wavlm         microsoft/wavlm-base-plus            speech SSL, 94M
  hubert        facebook/hubert-base-ls960           speech SSL, 94M
  wav2vec2      facebook/wav2vec2-base               speech SSL, 94M
  aves-bio      ESP AVES, HuBERT-style, bioacoustic  94M   (from GCS)

Every encoder is FROZEN. We extract per-layer hidden states, mean-pool over
time, and fit a logistic-regression probe with LEAVE-ONE-CAT-OUT splits.

Three numbers per (encoder, layer):
  ctx_loco   context accuracy, held-out cat        <- the real result
  ctx_rand   context accuracy, random 5-fold       <- the leaky number
  identity   cat-identity accuracy, random 5-fold  <- how much "who" is in there

The interesting quantity is not just the best ctx_loco. It is the ratio
ctx_loco / identity: an encoder that carries context *without* carrying
identity is the one that actually hears the situation.

Writes results.json + a layer-sweep figure. CPU only, ~10 min for all encoders.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

SR = 16_000
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_enc")
CAT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catmeows")
os.makedirs(OUT, exist_ok=True)


# ----------------------------------------------------------------- data
def load_clips():
    import soundfile as sf
    from scipy.signal import resample_poly

    files = sorted(glob.glob(os.path.join(CAT_DIR, "**", "*.wav"), recursive=True))
    wavs, meta = [], []
    for f in files:
        x, sr = sf.read(f, dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        if sr != SR:
            g = np.gcd(int(sr), SR)
            x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        if len(x) < 800:
            continue
        p = os.path.basename(f)[:-4].split("_")
        wavs.append(x)
        meta.append({"ctx": p[0], "cat": p[1], "breed": p[2], "sess": p[5] if len(p) > 5 else "?"})
    return wavs, meta


# ----------------------------------------------------------------- encoders
def hf_layers(model_id: str, wavs, max_s=6.0):
    """Return (n_layers+1, N, D) mean-pooled hidden states from a HF audio model."""
    import torch
    from transformers import AutoModel

    m = AutoModel.from_pretrained(model_id, output_hidden_states=True).eval()
    outs = []
    with torch.no_grad():
        for x in wavs:
            x = x[: int(max_s * SR)]
            if len(x) < 1600:
                x = np.pad(x, (0, 1600 - len(x)))
            # batch_size=1 on purpose: padding changes HuBERT-family embeddings
            t = torch.from_numpy(x).float().unsqueeze(0)
            t = (t - t.mean()) / (t.std() + 1e-7)
            hs = m(t).hidden_states                       # tuple(L+1) of (1,T,D)
            outs.append(np.stack([h[0].mean(0).numpy() for h in hs]))
    del m
    return np.stack(outs, axis=1)                          # (L+1, N, D)


def aves_layers(wavs, max_s=6.0):
    """ESP's AVES-bio: torchaudio wav2vec2 built from ESP's config + GCS weights."""
    import torch
    import torchaudio
    from urllib.request import urlretrieve

    base = "https://storage.googleapis.com/esp-public-files/ported_aves"
    cfg_p = os.path.join(OUT, "aves-base-bio.json")
    pt_p = os.path.join(OUT, "aves-base-bio.pt")
    if not os.path.exists(cfg_p):
        urlretrieve(f"{base}/aves-base-bio.torchaudio.model_config.json", cfg_p)
    if not os.path.exists(pt_p):
        print("  downloading AVES-bio weights (~378 MB)...")
        urlretrieve(f"{base}/aves-base-bio.torchaudio.pt", pt_p)

    cfg = json.load(open(cfg_p))
    m = torchaudio.models.wav2vec2_model(**cfg, aux_num_out=None)
    m.load_state_dict(torch.load(pt_p, map_location="cpu"))
    m.eval()

    outs = []
    with torch.no_grad():
        for x in wavs:
            x = x[: int(max_s * SR)]
            if len(x) < 1600:
                x = np.pad(x, (0, 1600 - len(x)))
            t = torch.from_numpy(x).float().unsqueeze(0)
            feats, _ = m.extract_features(t)               # list(L) of (1,T,D)
            outs.append(np.stack([f[0].mean(0).numpy() for f in feats]))
    del m
    return np.stack(outs, axis=1)


def mfcc_layers(wavs):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from context_probe import clip_vector
    return np.stack([clip_vector(x) for x in wavs])[None]  # (1, N, 85)


# ----------------------------------------------------------------- probes
def loco(X, y, grp, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    yp = np.empty_like(y)
    for g in np.unique(grp):
        te = grp == g
        sc = StandardScaler().fit(X[~te])
        yp[te] = LogisticRegression(max_iter=3000, C=C).fit(
            sc.transform(X[~te]), y[~te]).predict(sc.transform(X[te]))
    return yp


def kfold(X, y, C=1.0, k=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    yp = np.empty_like(y)
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=0).split(X, y):
        sc = StandardScaler().fit(X[tr])
        yp[te] = LogisticRegression(max_iter=3000, C=C).fit(
            sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    return yp


def evaluate(name, L, y, cats, res):
    """L: (n_layers, N, D). Sweep layers, record all three metrics."""
    from sklearn.metrics import balanced_accuracy_score
    ok = np.isin(cats, [c for c in np.unique(cats) if (cats == c).sum() >= 5])
    rows = []
    for li in range(L.shape[0]):
        X = L[li]
        ctx_loco = balanced_accuracy_score(y, loco(X, y, cats))
        ctx_rand = balanced_accuracy_score(y, kfold(X, y))
        ident = float((kfold(X[ok], cats[ok]) == cats[ok]).mean())
        rows.append({"layer": li, "dim": int(X.shape[1]),
                     "ctx_loco": ctx_loco, "ctx_rand": ctx_rand, "identity": ident})
        print(f"    L{li:<2d} d={X.shape[1]:<5d} ctx_loco {ctx_loco:.3f}  "
              f"ctx_rand {ctx_rand:.3f}  identity {ident:.3f}")
    best = max(rows, key=lambda r: r["ctx_loco"])
    res[name] = {"layers": rows, "best": best}
    print(f"  -> best layer {best['layer']} : ctx_loco {best['ctx_loco']:.3f} "
          f"(leaky {best['ctx_rand']:.3f}, identity {best['identity']:.3f})")
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)


def main():
    wavs, meta = load_clips()
    y = np.array([m["ctx"] for m in meta])
    cats = np.array([m["cat"] for m in meta])
    print(f"{len(wavs)} clips · {len(set(cats))} cats · "
          f"dur {np.mean([len(w)/SR for w in wavs]):.2f}s mean\n")

    res = {}
    if os.path.exists(os.path.join(OUT, "results.json")):
        res = json.load(open(os.path.join(OUT, "results.json")))

    jobs = [
        ("mfcc", lambda: mfcc_layers(wavs)),
        ("wavlm", lambda: hf_layers("microsoft/wavlm-base-plus", wavs)),
        ("hubert", lambda: hf_layers("facebook/hubert-base-ls960", wavs)),
        ("wav2vec2", lambda: hf_layers("facebook/wav2vec2-base", wavs)),
        ("aves-bio", lambda: aves_layers(wavs)),
    ]
    only = sys.argv[1:] or None
    for name, fn in jobs:
        if only and name not in only:
            continue
        if name in res:
            print(f"{name}: cached, skipping"); continue
        print(f"{name}:")
        try:
            L = fn()
            np.save(os.path.join(OUT, f"emb_{name}.npy"), L.astype(np.float16))
            evaluate(name, L, y, cats, res)
        except Exception as e:                              # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}")
        print()

    json.dump({"meta": meta}, open(os.path.join(OUT, "meta.json"), "w"))
    print("\n=== SUMMARY (chance 0.333) ===")
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]["best"]["ctx_loco"]):
        b = v["best"]
        print(f"  {k:<10s} L{b['layer']:<2d}  ctx {b['ctx_loco']:.3f}   "
              f"leaky {b['ctx_rand']:.3f}   identity {b['identity']:.3f}   "
              f"ctx/id {b['ctx_loco']/max(b['identity'],1e-6):.2f}")


if __name__ == "__main__":
    main()
