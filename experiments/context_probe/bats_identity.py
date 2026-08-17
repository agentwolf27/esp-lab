"""
E3-lite: identity decodability in a WILD species, at scale.

BEANS bats (Prat et al. 2017 via Earth Species Project): 10 Egyptian fruit
bats x 200 calls each (subsampled from 1000), recorded at 250 kHz.

Question 1: how decodable is individual identity from frozen encoders that
were built for 16 kHz human speech? If it is high here too, the identity-
leakage finding is not a domestic-animal / human-directed-vocalisation quirk.

Question 2 (ESP's own open problem #6 -- bandwidth): these calls carry energy
above 8 kHz that a 16 kHz encoder cannot see. Two ways in:
    native   resample 250k -> 16k       (keeps 0-8 kHz of the original)
    slow4x   relabel 250k as 62.5k, then -> 16k   (0-32 kHz folded into
             0-8 kHz; time stretched 4x; truncate at 6 s = 1.5 s original)
Does identity survive better under one or the other?

Identity is the label, so the split is plain stratified 5-fold (there is no
individual to hold out -- and no session metadata to hold out either, which
we note as a limitation).
"""
from __future__ import annotations

import glob, json, os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
D = os.path.dirname(os.path.abspath(__file__))
BD = os.path.join(D, "bats")
OUT = os.path.join(D, "out_bats"); os.makedirs(OUT, exist_ok=True)
SR = 16_000
MAX_S = 6.0


def load(mode):
    import soundfile as sf
    from scipy.signal import resample_poly
    sub = pd.read_csv(os.path.join(BD, "subset2000.csv"))
    wavs, lab = [], []
    for _, r in sub.iterrows():
        p = os.path.join(BD, "audio", r["File Name"])
        if not os.path.exists(p):
            continue
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        if mode == "native":
            src = sr                            # 250000
        else:                                   # slow4x: pretend it is 62.5 kHz
            src = sr // 4
        g = np.gcd(int(src), SR)
        x = resample_poly(x, SR // g, src // g).astype(np.float32)
        x = x[: int(MAX_S * SR)]
        if len(x) < 1600:
            x = np.pad(x, (0, 1600 - len(x)))
        wavs.append(x); lab.append(int(r["Emitter"]))
    return wavs, np.array(lab)


def embed_hf(model_id, wavs):
    import torch
    from transformers import AutoModel
    m = AutoModel.from_pretrained(model_id, output_hidden_states=True).eval()
    E = []
    with torch.no_grad():
        for i, x in enumerate(wavs):
            t = torch.from_numpy(x).float().unsqueeze(0)
            t = (t - t.mean()) / (t.std() + 1e-7)
            hs = m(t).hidden_states
            E.append(np.stack([h[0].mean(0).numpy() for h in hs]))
            if (i + 1) % 250 == 0:
                print(f"    {i+1}/{len(wavs)}", flush=True)
    return np.stack(E, axis=1)


def embed_aves(wavs):
    import torch, torchaudio
    cfg = json.load(open(os.path.join(D, "out_enc", "aves-base-bio.json")))
    m = torchaudio.models.wav2vec2_model(**cfg, aux_num_out=None)
    m.load_state_dict(torch.load(os.path.join(D, "out_enc", "aves-base-bio.pt"), map_location="cpu"))
    m.eval(); E = []
    with torch.no_grad():
        for i, x in enumerate(wavs):
            f, _ = m.extract_features(torch.from_numpy(x).float().unsqueeze(0))
            E.append(np.stack([h[0].mean(0).numpy() for h in f]))
            if (i + 1) % 250 == 0:
                print(f"    {i+1}/{len(wavs)}", flush=True)
    return np.stack(E, axis=1)


def identity_acc(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    yp = np.empty_like(y)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        sc = StandardScaler().fit(X[tr])
        yp[te] = LogisticRegression(max_iter=3000, C=0.5).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    return float((yp == y).mean())


def main():
    res = {}
    for mode in ["native", "slow4x"]:
        wavs, y = load(mode)
        print(f"\n=== {mode}: {len(wavs)} calls, {len(set(y))} bats, "
              f"mean {np.mean([len(w)/SR for w in wavs]):.2f}s ===")
        for name, fn in [("wavlm", lambda: embed_hf("microsoft/wavlm-base-plus", wavs)),
                         ("aves-bio", lambda: embed_aves(wavs))]:
            key = f"{name}_{mode}"
            cache = os.path.join(OUT, f"emb_{key}.npy")
            if os.path.exists(cache):
                E = np.load(cache).astype(np.float32); print(f"  {name}: cached")
            else:
                print(f"  {name}: embedding...", flush=True)
                E = fn(); np.save(cache, E.astype(np.float16))
            accs = [identity_acc(E[li], y) for li in range(E.shape[0])]
            best = int(np.argmax(accs))
            res[key] = {"per_layer": accs, "best_layer": best, "best": accs[best],
                        "mean": float(np.mean(accs))}
            print(f"  {name:<9s} identity acc  best L{best} {accs[best]:.3f}   "
                  f"mean {np.mean(accs):.3f}   (chance 0.100)")
            print(f"           per-layer: {' '.join(f'{a:.2f}' for a in accs)}")
            json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)

    print("\n=== SUMMARY (10 wild bats, chance 0.10) ===")
    for k, v in res.items():
        print(f"  {k:<18s} best {v['best']:.3f} (L{v['best_layer']})   mean {v['mean']:.3f}")


if __name__ == "__main__":
    main()
