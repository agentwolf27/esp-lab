"""
E5 -- channel stress test. The thesis question in miniature.

Deployment never matches the recording session: a different mic, a bigger
room, more distance, wind. Take the SAME 656 cat/dog clips, perturb them four
ways, re-embed with WavLM, and ask two probes trained on CLEAN audio:

    context probe   (isolation vs brushing / aggression vs play)
    identity probe  (which animal, 20 cats / 10 dogs)

For each perturbation, measure the fraction of decisions that FLIP relative
to the clean prediction, and the accuracy under perturbation. Two outcomes:

    context stable, identity fragile  -> the affect axis is channel-robust,
                                         identity is channel-entangled (good
                                         news for anything built on context;
                                         a lovely trustworthiness result)
    context flips as much as identity -> the "context" the probe found was
                                         channel all along (consistent with
                                         the room finding, bad news for
                                         claims of context decoding)

Perturbations (all cheap, all realistic):
    lowpass4k   4 kHz low-pass          (cheap mic / phone / distant source)
    reverb      synthetic room IR       (bigger room, ~0.4 s decay)
    gain-12     -12 dB                  (further away)
    noise+10    +10 dB SNR pink-ish noise (wind / traffic floor)

Runtime: 656 clips x 4 perturbations x 1 encoder ~ 15-20 min CPU.
"""
from __future__ import annotations

import json, os, warnings
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve

warnings.filterwarnings("ignore")
D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "out_stress"); os.makedirs(OUT, exist_ok=True)
SR = 16_000


def lowpass(x, fc=4000):
    sos = butter(6, fc / (SR / 2), btype="low", output="sos")
    return sosfilt(sos, x).astype(np.float32)


def reverb(x, rt60=0.4, seed=0):
    rng = np.random.default_rng(seed)
    n = int(rt60 * SR)
    ir = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9 * np.arange(n) / n)  # -60 dB at rt60
    ir[0] = 1.0
    ir /= np.sqrt((ir ** 2).sum())
    y = fftconvolve(x, ir)[: len(x)]
    return (y / (np.abs(y).max() + 1e-8) * np.abs(x).max()).astype(np.float32)


def gain(x, db=-12):
    return (x * 10 ** (db / 20)).astype(np.float32)


def add_noise(x, snr_db=10, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(len(x)).astype(np.float32)
    # pink-ish: integrate white then high-pass lightly
    p = np.cumsum(w); p -= np.convolve(p, np.ones(400) / 400, mode="same")
    p /= (p.std() + 1e-8)
    ps = (x ** 2).mean(); pn = ps / (10 ** (snr_db / 10))
    return (x + p * np.sqrt(pn)).astype(np.float32)


PERTURB = {"clean": lambda x: x, "lowpass4k": lowpass, "reverb": reverb,
           "gain-12": gain, "noise+10dB": add_noise}


def embed(m, wavs):
    import torch
    E = []
    with torch.no_grad():
        for x in wavs:
            t = torch.from_numpy(x).float().unsqueeze(0)
            t = (t - t.mean()) / (t.std() + 1e-7)
            hs = m(t).hidden_states
            E.append(np.stack([h[0].mean(0).numpy() for h in hs]))
    return np.stack(E, axis=1)


def fit(Xa, ya):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(Xa, ya)


def bacc(a, b):
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(a, b)


def main():
    from transformers import AutoModel
    import cross_species as cs
    rows = cs.load_all()
    sp = np.array([r["species"] for r in rows]); y = np.array([r["pol"] for r in rows])
    ind = np.array([r["indiv"] for r in rows])
    m = AutoModel.from_pretrained("microsoft/wavlm-base-plus", output_hidden_states=True).eval()

    embs = {}
    for name, fn in PERTURB.items():
        cache = os.path.join(OUT, f"emb_{name}.npy")
        if os.path.exists(cache):
            embs[name] = np.load(cache).astype(np.float32); print(f"{name}: cached")
        else:
            print(f"{name}: embedding...", flush=True)
            E = embed(m, [fn(r["wav"]) for r in rows]); np.save(cache, E.astype(np.float16)); embs[name] = E
    L = embs["clean"].shape[0]
    LAYERS = [3, 6, 9, 12]

    res = {}
    print(f"\n{'perturbation':<12s} {'layer':>5} | {'ctx acc':>7} {'ctx flip':>8} | {'id acc':>7} {'id flip':>7}")
    for name in PERTURB:
        if name == "clean":
            continue
        for li in LAYERS:
            row = {}
            for spc in ["cat", "dog"]:
                msk = sp == spc
                Xc = embs["clean"][li][msk]; Xp = embs[name][li][msk]
                mu, sd = Xc.mean(0), Xc.std(0) + 1e-8          # scaler from CLEAN only
                Zc, Zp = (Xc - mu) / sd, (Xp - mu) / sd
                # context probe: train on clean, all animals (in-sample -- we care about flips, not generalisation)
                pc = fit(Zc, y[msk]); ctx_clean = pc.predict(Zc); ctx_pert = pc.predict(Zp)
                pi = fit(Zc, ind[msk]); id_clean = pi.predict(Zc); id_pert = pi.predict(Zp)
                row[spc] = {"ctx_acc_clean": bacc(y[msk], ctx_clean), "ctx_acc_pert": bacc(y[msk], ctx_pert),
                            "ctx_flip": float((ctx_clean != ctx_pert).mean()),
                            "id_acc_clean": float((id_clean == ind[msk]).mean()),
                            "id_acc_pert": float((id_pert == ind[msk]).mean()),
                            "id_flip": float((id_clean != id_pert).mean())}
            # pooled summary
            ctx_flip = np.mean([row[s]["ctx_flip"] for s in row]); id_flip = np.mean([row[s]["id_flip"] for s in row])
            ctx_acc = np.mean([row[s]["ctx_acc_pert"] for s in row]); id_acc = np.mean([row[s]["id_acc_pert"] for s in row])
            res[f"{name}_L{li}"] = row
            print(f"{name:<12s} {li:>5} | {ctx_acc:>7.3f} {ctx_flip:>8.3f} | {id_acc:>7.3f} {id_flip:>7.3f}")

    # summary at the deep layer
    print("\n=== deep layer (L9) summary, pooled cats+dogs ===")
    summ = {}
    for name in PERTURB:
        if name == "clean": continue
        r = res[f"{name}_L9"]
        cf = np.mean([r[s]["ctx_flip"] for s in r]); idf = np.mean([r[s]["id_flip"] for s in r])
        summ[name] = {"ctx_flip": float(cf), "id_flip": float(idf),
                      "ctx_acc": float(np.mean([r[s]["ctx_acc_pert"] for s in r])),
                      "id_acc": float(np.mean([r[s]["id_acc_pert"] for s in r]))}
        print(f"  {name:<12s} context flips {cf:.1%}   identity flips {idf:.1%}   ratio id/ctx {idf/max(cf,1e-6):.1f}x")
    json.dump({"per_layer": res, "summary_L9": summ}, open(os.path.join(OUT, "results.json"), "w"), indent=1)

    print("\n=== VERDICT ===")
    ratios = [summ[k]["id_flip"] / max(summ[k]["ctx_flip"], 1e-6) for k in summ]
    if min(ratios) > 1.5:
        print("  identity is consistently MORE fragile than context under channel change")
        print("  -> the context axis is the channel-robust one. Good news for anything built on it.")
    elif max(ratios) < 1.2:
        print("  context flips about as much as identity: the 'context' the probe uses is channel-entangled.")
    else:
        print("  mixed: depends on the perturbation. Report per-perturbation.")


if __name__ == "__main__":
    main()
