"""
Experiment 3: does an affect axis learned on one species transfer to another?

This is the rigorous version of "can a dog and a cat talk to each other".
It does NOT ask whether they share words. It asks a testable question:

    Is the acoustic difference between a distressed call and a contented call
    the SAME DIRECTION in embedding space for cats and for dogs?

If yes, a probe trained only on cats predicts dog contexts above chance,
without ever seeing a dog. That would be evidence for shared acoustic
structure across species -- the "acoustic universals of arousal" hypothesis
(Morton 1977; Briefer 2012; Filippi 2017) tested inside a neural embedding.

Binary axis, chosen to be defensible in both species:
    NEGATIVE / agonistic-distress :  cat isolation      | dog aggression
    POSITIVE / affiliative        :  cat brushing       | dog play
Excluded as ambiguous: cat "waiting for food" (anticipation, unclear valence),
dog "contact" (neutral//varied).

Protocol
    - frozen encoder, mean-pooled hidden states, one layer at a time
    - z-score WITHIN species before transfer (removes species-level offset;
      without this you are just measuring "is it a cat or a dog")
    - within-species score = leave-one-INDIVIDUAL-out
    - transfer score = train on all of species A, test on all of species B
    - control = shuffled labels, to confirm chance is where we think it is
"""
from __future__ import annotations

import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "out_cross")
os.makedirs(OUT, exist_ok=True)
SR = 16_000
MAX_S = 6.0

# label -> (species, polarity)  polarity: 0 = positive/affiliative, 1 = negative/distress
CAT_MAP = {"I": 1, "B": 0}
DOG_MAP = {"aggression": 1, "play": 0}


def _read(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x[: int(MAX_S * SR)]


def load_all():
    rows = []
    for f in sorted(glob.glob(os.path.join(D, "catmeows", "**", "*.wav"), recursive=True)):
        p = os.path.basename(f)[:-4].split("_")
        if p[0] not in CAT_MAP:
            continue
        rows.append({"path": f, "species": "cat", "ctx": p[0],
                     "pol": CAT_MAP[p[0]], "indiv": "cat_" + p[1]})
    ann = pd.read_csv(os.path.join(D, "dogs", "annotations.csv"))
    for _, r in ann.iterrows():
        if r["context"] not in DOG_MAP:
            continue
        rows.append({"path": os.path.join(D, "dogs", "audio", r["filename"]),
                     "species": "dog", "ctx": r["context"],
                     "pol": DOG_MAP[r["context"]], "indiv": "dog_" + r["name"]})
    out = []
    for r in rows:
        try:
            x = _read(r["path"])
        except Exception as e:                                   # noqa: BLE001
            print("skip", os.path.basename(r["path"]), e); continue
        if len(x) < 1600:
            x = np.pad(x, (0, 1600 - len(x)))
        r["wav"] = x
        out.append(r)
    return out


def embed(rows, model_id="microsoft/wavlm-base-plus"):
    import torch
    from transformers import AutoModel
    m = AutoModel.from_pretrained(model_id, output_hidden_states=True).eval()
    E = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            t = torch.from_numpy(r["wav"]).float().unsqueeze(0)
            t = (t - t.mean()) / (t.std() + 1e-7)
            hs = m(t).hidden_states
            E.append(np.stack([h[0].mean(0).numpy() for h in hs]))
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(rows)}")
    return np.stack(E, axis=1)                                    # (L+1, N, D)


def zscore_within(X, species):
    """Remove the species-level offset. Without this, cross-species 'transfer'
    is partly just detecting which species it is."""
    Z = X.copy()
    for s in np.unique(species):
        m = species == s
        Z[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-8)
    return Z


def loio(X, y, indiv):
    """leave-one-individual-out"""
    from sklearn.linear_model import LogisticRegression
    yp = np.empty_like(y)
    for g in np.unique(indiv):
        te = indiv == g
        if len(np.unique(y[~te])) < 2:
            yp[te] = y[~te][0] if (~te).sum() else 0
            continue
        yp[te] = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(
            X[~te], y[~te]).predict(X[te])
    return yp


def transfer(Xa, ya, Xb):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(Xa, ya).predict(Xb)


def main():
    from sklearn.metrics import balanced_accuracy_score as bacc

    rows = load_all()
    sp = np.array([r["species"] for r in rows])
    y = np.array([r["pol"] for r in rows])
    ind = np.array([r["indiv"] for r in rows])
    print(f"{len(rows)} clips")
    for s in ["cat", "dog"]:
        m = sp == s
        print(f"  {s}: {m.sum():4d}  positive={int((y[m]==0).sum()):3d}  "
              f"negative={int((y[m]==1).sum()):3d}  individuals={len(set(ind[m]))}")

    cache = os.path.join(OUT, "emb_wavlm.npy")
    if os.path.exists(cache):
        E = np.load(cache).astype(np.float32); print("cached embeddings")
    else:
        print("embedding with WavLM-base-plus...")
        E = embed(rows)
        np.save(cache, E.astype(np.float16))
    json.dump([{k: v for k, v in r.items() if k != "wav"} for r in rows],
              open(os.path.join(OUT, "meta.json"), "w"))

    res = []
    rng = np.random.default_rng(0)
    for li in range(E.shape[0]):
        Z = zscore_within(E[li], sp)
        c, d = sp == "cat", sp == "dog"
        r = {"layer": li}
        r["cat_within"] = bacc(y[c], loio(Z[c], y[c], ind[c]))
        r["dog_within"] = bacc(y[d], loio(Z[d], y[d], ind[d]))
        r["cat_to_dog"] = bacc(y[d], transfer(Z[c], y[c], Z[d]))
        r["dog_to_cat"] = bacc(y[c], transfer(Z[d], y[d], Z[c]))
        # control: shuffle the training labels; should sit at 0.5
        r["cat_to_dog_shuf"] = bacc(y[d], transfer(Z[c], rng.permutation(y[c]), Z[d]))
        r["dog_to_cat_shuf"] = bacc(y[c], transfer(Z[d], rng.permutation(y[d]), Z[c]))
        res.append(r)
        print(f"  L{li:<2d} cat_within {r['cat_within']:.3f}  dog_within {r['dog_within']:.3f}  "
              f"| cat->dog {r['cat_to_dog']:.3f}  dog->cat {r['dog_to_cat']:.3f}  "
              f"| shuf {r['cat_to_dog_shuf']:.3f}/{r['dog_to_cat_shuf']:.3f}")

    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    best_t = max(res, key=lambda r: (r["cat_to_dog"] + r["dog_to_cat"]) / 2)
    print(f"\nbest transfer layer L{best_t['layer']}: "
          f"cat->dog {best_t['cat_to_dog']:.3f}, dog->cat {best_t['dog_to_cat']:.3f}, "
          f"mean {(best_t['cat_to_dog']+best_t['dog_to_cat'])/2:.3f}  (chance 0.500)")
    print(f"shuffled control mean "
          f"{(best_t['cat_to_dog_shuf']+best_t['dog_to_cat_shuf'])/2:.3f}")


if __name__ == "__main__":
    main()
