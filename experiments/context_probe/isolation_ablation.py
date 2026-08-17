"""
Why do dog->cat and pig->cat both work, while nothing transfers INTO dog or pig?

Hypothesis: the shared axis is separation/isolation distress. Cat negatives ARE
isolation calls; pig negatives are dominated by piglet isolation (2048 of 4572,
capped at 700 here). Young-mammal isolation calls are famously similar across
species (the "distress cry" story). If that is what transfers, then:

  (1) pig->cat should COLLAPSE when pig isolation is removed from training
  (2) pig->cat should be as strong or stronger when the source is ONLY
      isolation-vs-positive
  (3) dog->cat should not depend on this (dogs have no isolation class) --
      so if dog->cat holds and pig->cat collapses, there are two different
      shared axes, not one.

Also: per-context breakdown of the pig-trained probe on cat targets, and of the
cat-trained probe on each PIG context (which pig contexts does a cat "distress"
probe flag as negative?). All from cached WavLM embeddings. ~3 minutes.
"""
from __future__ import annotations

import json, os, warnings
import numpy as np

warnings.filterwarnings("ignore")
D = os.path.dirname(os.path.abspath(__file__))


def fit(Xa, ya):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(Xa, ya)


def bacc(a, b):
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(a, b)


def zgroup(X, g):
    Z = X.copy()
    for s in np.unique(g):
        m = g == s
        Z[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-8)
    return Z


def main():
    import cross_species as cs
    rows = cs.load_all()
    Ecd = np.load(os.path.join(D, "out_cross", "emb_wavlm.npy")).astype(np.float32)
    sp = np.array([r["species"] for r in rows]); y = np.array([r["pol"] for r in rows])
    ctx_cd = np.array([r["ctx"] for r in rows])
    Ep = np.load(os.path.join(D, "out_pigs", "emb_wavlm.npy")).astype(np.float32)
    mp = json.load(open(os.path.join(D, "out_pigs", "meta.json")))
    yp = np.array([m["pol"] for m in mp]); team = np.array([m["team"] for m in mp])
    ctxp = np.array([m["ctx"] for m in mp])
    L = Ep.shape[0]
    c, d = sp == "cat", sp == "dog"
    Ec, yc = Ecd[:, c], y[c]
    Ed, yd = Ecd[:, d], y[d]
    rng = np.random.default_rng(3)
    out = {}

    def transfer(Esrc, ysrc, gsrc, Etgt, ytgt, gtgt, n_perm=200):
        accs, null = [], []
        Zs = [zgroup(Esrc[li], gsrc) for li in range(L)]
        Zt = [zgroup(Etgt[li], gtgt) for li in range(L)]
        for li in range(L):
            accs.append(bacc(ytgt, fit(Zs[li], ysrc).predict(Zt[li])))
        obs = float(np.mean(accs))
        for _ in range(n_perm):
            yperm = ysrc.copy()
            for g in np.unique(gsrc):
                m = gsrc == g; yperm[m] = rng.permutation(ysrc[m])
            null.append(np.mean([bacc(ytgt, fit(Zs[li], yperm).predict(Zt[li])) for li in range(L)]))
        p = (np.sum(np.array(null) >= obs) + 1) / (n_perm + 1)
        return obs, float(p), accs

    zero_c = np.zeros(len(yc)); zero_d = np.zeros(len(yd))
    print("=== pig -> cat, by which pig contexts are in the SOURCE (mean over layers, chance 0.5) ===")
    variants = {
        "all pig contexts (as before)": np.ones(len(yp), bool),
        "pig WITHOUT isolation": ctxp != "Isolation",
        "pig isolation vs all positive ONLY": (ctxp == "Isolation") | (yp == 0),
        "pig negatives EXCEPT isolation vs positive": (ctxp != "Isolation"),
    }
    for name, m in variants.items():
        if len(np.unique(yp[m])) < 2:
            continue
        obs, p, _ = transfer(Ep[:, m], yp[m], team[m], Ec, yc, zero_c, n_perm=100)
        out[name] = {"acc": obs, "p": p, "n": int(m.sum())}
        print(f"  {name:<44s} {obs:.3f}  p={p:.3f}  (n={m.sum()})")

    print("\n=== dog -> cat for reference (dogs have no isolation class) ===")
    obs, p, _ = transfer(Ed, yd, zero_d, Ec, yc, zero_c, n_perm=100)
    out["dog_to_cat"] = {"acc": obs, "p": p}
    print(f"  dog -> cat  {obs:.3f}  p={p:.3f}")

    print("\n=== which PIG contexts does a CAT-trained (isolation vs brushing) probe call 'negative'? ===")
    # train on cats, score every pig context: fraction predicted negative (mean over layers)
    frac = {}
    for li in range(L):
        Zc = zgroup(Ec[li], zero_c); Zp = zgroup(Ep[li], team)
        pred = fit(Zc, yc).predict(Zp)
        for cx in np.unique(ctxp):
            frac.setdefault(cx, []).append(float(pred[ctxp == cx].mean()))
    print(f"  {'pig context':<22s} {'valence':>7} {'frac called NEG by cat probe':>30}")
    valence = {m["ctx"]: ("Neg" if m["pol"] == 1 else "Pos") for m in mp}
    for cx in sorted(frac, key=lambda k: -np.mean(frac[k])):
        print(f"  {cx:<22s} {valence[cx]:>7} {np.mean(frac[cx]):>30.3f}")
    out["cat_probe_on_pig_contexts"] = {k: float(np.mean(v)) for k, v in frac.items()}

    print("\n=== which CAT contexts does a PIG-trained probe call 'negative'? (incl. excluded 'F' food) ===")
    # need all cat clips including waiting-for-food -> reload from catmeows via context_probe loader
    import glob, soundfile as sf
    from scipy.signal import resample_poly
    files = sorted(glob.glob(os.path.join(D, "catmeows", "**", "*.wav"), recursive=True))
    # only re-embed the F clips (not cached): 92 clips, ~30s
    F = [f for f in files if os.path.basename(f).startswith("F_")]
    import torch
    from transformers import AutoModel
    mdl = AutoModel.from_pretrained("microsoft/wavlm-base-plus", output_hidden_states=True).eval()
    EF = []
    with torch.no_grad():
        for f in F:
            x, sr = sf.read(f, dtype="float32")
            if x.ndim > 1: x = x.mean(1)
            if sr != 16000:
                g = np.gcd(int(sr), 16000); x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
            x = x[:96000]
            if len(x) < 1600: x = np.pad(x, (0, 1600 - len(x)))
            t = torch.from_numpy(x).float().unsqueeze(0); t = (t - t.mean()) / (t.std() + 1e-7)
            EF.append(np.stack([h[0].mean(0).numpy() for h in mdl(t).hidden_states]))
    EF = np.stack(EF, axis=1)
    Ecat_all = np.concatenate([Ec, EF], axis=1)
    ctx_all = np.concatenate([ctx_cd[c], np.array(["F"] * EF.shape[1])])
    fr = {}
    for li in range(L):
        Zp = zgroup(Ep[li], team); Zc = zgroup(Ecat_all[li], np.zeros(Ecat_all.shape[1]))
        pred = fit(Zp, yp).predict(Zc)
        for cx in ["B", "F", "I"]:
            fr.setdefault(cx, []).append(float(pred[ctx_all == cx].mean()))
    names = {"B": "brushing (pos)", "F": "waiting for food (excluded)", "I": "isolation (neg)"}
    for cx in ["B", "F", "I"]:
        print(f"  {names[cx]:<30s} frac called NEG by pig probe: {np.mean(fr[cx]):.3f}")
    out["pig_probe_on_cat_contexts"] = {names[k]: float(np.mean(v)) for k, v in fr.items()}

    json.dump(out, open(os.path.join(D, "out_pigs", "isolation_ablation.json"), "w"), indent=1)
    print("\nwrote out_pigs/isolation_ablation.json")


if __name__ == "__main__":
    main()
