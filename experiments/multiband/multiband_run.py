"""
Does ESP's adaptive multi-band encoding beat our time-expansion baseline on
10-way Egyptian-fruit-bat IDENTITY?

Conditions (all on the SAME 1,000-call subset = 100 calls x 10 emitters):
  baseband      250 kHz -> 16 kHz resample                     (0-8 kHz kept)
  timeexp4x     relabel 250 kHz as 62.5 kHz -> 16 kHz          (0-32 kHz folded, 4x slower)
  multiband     mba.MultibandTransform(sr=250k, max_freq=32k)  -> 4 bands x 16 kHz
                fused by (a) mean over bands, (b) concat of 4 band embeddings

Encoders: microsoft/wavlm-base-plus, AVES-bio. Frozen, mean-pooled hidden states.
Probe: identity_acc from bats_identity.py (stratified 5-fold logistic regression).
"""
from __future__ import annotations

import torch
torch.set_num_threads(2)

import glob, json, os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from bats_identity import identity_acc          # reuse the exact probe

BD = os.path.join(D, "bats")
OUT = os.path.join(D, "out_multiband")
os.makedirs(OUT, exist_ok=True)

SR = 16_000
NATIVE_SR = 250_000
MAX_S = 6.0
PER_EMITTER = 100
MAX_FREQ = 32_000                                # 4 bands of 8 kHz
NBANDS = 4


# ---------------------------------------------------------------- stage 1
def build_waveforms():
    """One pass over the 250 kHz WAVs -> baseband, time-expansion, 4 bands."""
    import soundfile as sf
    from scipy.signal import resample_poly
    import multiband_audio as mba

    meta_p = os.path.join(OUT, "meta.json")
    if os.path.exists(meta_p):
        print("stage1: cached")
        return json.load(open(meta_p))

    sub = pd.read_csv(os.path.join(BD, "subset2000.csv"))
    sub = sub.groupby("Emitter", sort=True).head(PER_EMITTER).reset_index(drop=True)
    rows = [r for _, r in sub.iterrows()
            if os.path.exists(os.path.join(BD, "audio", r["File Name"]))]
    print(f"stage1: {len(rows)} calls, {sub['Emitter'].nunique()} emitters")

    # pre-size from headers so we can fill memmaps in place (8 GB RAM budget)
    n_nat, n_slow = [], []
    for r in rows:
        info = sf.info(os.path.join(BD, "audio", r["File Name"]))
        f = info.frames
        n_nat.append(min(int(np.ceil(f / NATIVE_SR * SR)), int(MAX_S * SR)))
        n_slow.append(min(int(np.ceil(f / (NATIVE_SR // 4) * SR)), int(MAX_S * SR)))
    n_nat, n_slow = np.array(n_nat), np.array(n_slow)
    print(f"  native total {n_nat.sum()/SR/60:.1f} min, slow4x total {n_slow.sum()/SR/60:.1f} min")

    off_n = np.concatenate([[0], np.cumsum(n_nat)])
    off_s = np.concatenate([[0], np.cumsum(n_slow)])
    W_nat = np.lib.format.open_memmap(os.path.join(OUT, "wav_baseband.npy"), mode="w+",
                                      dtype=np.float16, shape=(int(off_n[-1]),))
    W_slow = np.lib.format.open_memmap(os.path.join(OUT, "wav_timeexp4x.npy"), mode="w+",
                                       dtype=np.float16, shape=(int(off_s[-1]),))
    W_band = np.lib.format.open_memmap(os.path.join(OUT, "wav_bands.npy"), mode="w+",
                                       dtype=np.float16, shape=(NBANDS, int(off_n[-1])))

    tf = mba.MultibandTransform(sample_rate=NATIVE_SR, target_sr=SR, max_freq=MAX_FREQ)
    bands_hz = tf.get_band_info()
    print(f"  bands: {bands_hz}")

    labels, names = [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        x, sr = sf.read(os.path.join(BD, "audio", r["File Name"]), dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        assert sr == NATIVE_SR, sr

        # --- baseband: identical to bats_identity.load("native")
        g = np.gcd(NATIVE_SR, SR)
        a = resample_poly(x, SR // g, NATIVE_SR // g).astype(np.float32)[: int(MAX_S * SR)]
        # --- time expansion 4x: identical to bats_identity.load("slow4x")
        src = NATIVE_SR // 4
        g = np.gcd(src, SR)
        b = resample_poly(x, SR // g, src // g).astype(np.float32)[: int(MAX_S * SR)]
        # --- multiband: truncate native to 6 s real time, then heterodyne
        xt = torch.from_numpy(x[: int(MAX_S * NATIVE_SR)]).float().unsqueeze(0)
        bb = tf(xt)[0].numpy()                                    # (4, T')

        # store at unit std: float16 has no headroom at rms ~4e-4 (band 3), and
        # the encoders z-score every input anyway, so this changes nothing.
        u = lambda v: (v / (v.std() + 1e-12)).astype(np.float16)

        na, ns = n_nat[i], n_slow[i]
        W_nat[off_n[i]:off_n[i] + na] = u(a[:na])
        W_slow[off_s[i]:off_s[i] + ns] = u(b[:ns])
        m = min(bb.shape[1], na)
        W_band[:, off_n[i]:off_n[i] + m] = np.stack([u(bb[k, :m]) for k in range(NBANDS)])
        labels.append(int(r["Emitter"])); names.append(r["File Name"])

        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"    {i+1}/{len(rows)}  {el:.0f}s  eta {el/(i+1)*(len(rows)-i-1):.0f}s", flush=True)

    W_nat.flush(); W_slow.flush(); W_band.flush()
    meta = {"n": len(rows), "labels": labels, "files": names,
            "off_baseband": off_n.tolist(), "off_timeexp4x": off_s.tolist(),
            "off_bands": off_n.tolist(), "bands_hz": bands_hz,
            "per_emitter": PER_EMITTER, "max_s": MAX_S, "sr": SR}
    json.dump(meta, open(meta_p, "w"))
    return meta


def iter_wavs(meta, cond):
    """Yield float32 waveforms for cond in {baseband, timeexp4x} or band k."""
    if cond.startswith("band"):
        k = int(cond[4:])
        W = np.load(os.path.join(OUT, "wav_bands.npy"), mmap_mode="r")
        off = meta["off_bands"]
        for i in range(meta["n"]):
            yield np.asarray(W[k, off[i]:off[i + 1]], dtype=np.float32)
    else:
        W = np.load(os.path.join(OUT, f"wav_{cond}.npy"), mmap_mode="r")
        off = meta[f"off_{cond}"]
        for i in range(meta["n"]):
            yield np.asarray(W[off[i]:off[i + 1]], dtype=np.float32)


# ---------------------------------------------------------------- stage 2
def embed_all(meta, enc):
    """Embed every condition with ONE encoder (loaded once). -> dict cond -> (L,N,D)."""
    conds = ["baseband", "timeexp4x"] + [f"band{k}" for k in range(NBANDS)]
    todo = [c for c in conds if not os.path.exists(os.path.join(OUT, f"emb_{enc}_{c}.npy"))]
    if not todo:
        print(f"stage2 {enc}: all cached")
        return

    if enc == "wavlm":
        from transformers import AutoModel
        m = AutoModel.from_pretrained("microsoft/wavlm-base-plus", output_hidden_states=True).eval()
        def feats(t):
            return m(t).hidden_states
    else:
        import torchaudio
        cfg = json.load(open(os.path.join(D, "out_enc", "aves-base-bio.json")))
        m = torchaudio.models.wav2vec2_model(**cfg, aux_num_out=None)
        m.load_state_dict(torch.load(os.path.join(D, "out_enc", "aves-base-bio.pt"), map_location="cpu"))
        m.eval()
        def feats(t):
            return m.extract_features(t)[0]

    for cond in todo:
        print(f"stage2 {enc}/{cond}: embedding...", flush=True)
        E, t0 = [], time.time()
        with torch.no_grad():
            for i, x in enumerate(iter_wavs(meta, cond)):
                if len(x) < 1600:
                    x = np.pad(x, (0, 1600 - len(x)))
                t = torch.from_numpy(x).float().unsqueeze(0)
                t = (t - t.mean()) / (t.std() + 1e-7)          # same as embed_hf
                E.append(np.stack([h[0].mean(0).numpy() for h in feats(t)]))
                if (i + 1) % 250 == 0:
                    el = time.time() - t0
                    print(f"    {i+1}/{meta['n']}  {el:.0f}s", flush=True)
        np.save(os.path.join(OUT, f"emb_{enc}_{cond}.npy"),
                np.stack(E, axis=1).astype(np.float16))         # (L, N, D)
        del E
    del m


# ---------------------------------------------------------------- stage 3
def probe(meta, enc, res):
    y = np.array(meta["labels"])
    load = lambda c: np.load(os.path.join(OUT, f"emb_{enc}_{c}.npy")).astype(np.float32)

    base = load("baseband")
    texp = load("timeexp4x")
    bands = np.stack([load(f"band{k}") for k in range(NBANDS)])   # (B, L, N, D)
    L = base.shape[0]

    sets = {
        "baseband":          [base[li] for li in range(L)],
        "timeexp4x":         [texp[li] for li in range(L)],
        "multiband_mean":    [bands[:, li].mean(0) for li in range(L)],
        "multiband_concat":  [np.concatenate(list(bands[:, li]), axis=1) for li in range(L)],
    }
    for k in range(NBANDS):
        sets[f"multiband_band{k}_only"] = [bands[k, li] for li in range(L)]

    for name, Xs in sets.items():
        t0 = time.time()
        accs = [identity_acc(X, y) for X in Xs]
        bi = int(np.argmax(accs))
        sel = [3, 6, 9, 12] if L > 12 else [3, 6, 9, 11]
        sel = [s for s in sel if s < L]
        bs = max(sel, key=lambda s: accs[s])
        res[f"{enc}_{name}"] = {
            "per_layer": accs, "best_layer": bi, "best": accs[bi],
            "mean": float(np.mean(accs)),
            "layers_3_6_9_12": {str(s): accs[s] for s in sel},
            "best_of_3_6_9_12": accs[bs], "best_layer_of_3_6_9_12": bs,
        }
        print(f"  {enc:<9s} {name:<24s} best L{bi} {accs[bi]:.3f} | "
              f"L{{3,6,9,12}} best L{bs} {accs[bs]:.3f} | mean {np.mean(accs):.3f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)


def main():
    meta = build_waveforms()
    res = {}
    rp = os.path.join(OUT, "results.json")
    if os.path.exists(rp):
        res = json.load(open(rp))
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for enc in (["wavlm", "aves-bio"] if only is None else [only]):
        embed_all(meta, enc)
        print(f"\n=== probe {enc} (n={meta['n']}, chance 0.100) ===")
        probe(meta, enc, res)
    print("\n=== SUMMARY ===")
    for k, v in res.items():
        print(f"  {k:<40s} best {v['best']:.3f} (L{v['best_layer']})")


if __name__ == "__main__":
    main()
