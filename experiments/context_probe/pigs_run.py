"""
Third species: pigs (Soundwel / Briefer et al. 2022, CC-BY-4.0).

6,887 calls, 17 contexts, each context wholly positive or negative valence,
recorded by SIX different labs. That last fact is the point: valence is
confounded with recording team, so a random split would learn the lab.
The honest unit here is the LAB -- leave-one-team-out -- which is exactly the
site-shift protocol the whole thesis is about.

Part A  within-pig valence
        - frozen WavLM, per layer, leave-one-team-out (teams with both classes)
        - vs random 5-fold (leaky)
        - team decodability (6-way): how much "which lab" is in the embedding
        - the paper's own 19 acoustic features (in the key) as the published baseline
        - duration alone
Part B  cross-species transfer, 3x3 matrix (cat / dog / pig), mean over layers,
        z-scored within species (and, for pigs, within team), with the
        placebo (pig SEX -> cat AFFECT) and permutation nulls at the right unit.

Subsampled to keep the laptop honest: cap 700 calls per context (~4.9k total).
"""
from __future__ import annotations

import json, os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
D = os.path.dirname(os.path.abspath(__file__))
PD = os.path.join(D, "pigs")
OUT = os.path.join(D, "out_pigs"); os.makedirs(OUT, exist_ok=True)
SR = 16_000
MAX_S = 6.0
CAP = 700
ACOUSTIC = ["F0Mean", "Q25", "Q50", "Q75", "Fpeak", "Dur", "AMVar", "AMRate", "AMExtent",
            "Harmonicity", "WienEntrMean", "DFMin", "LPC1", "LPC2", "LPC3", "LPC4", "LPC5", "LPC6"]


def load_pigs():
    import soundfile as sf
    from scipy.signal import resample_poly
    key = pd.read_excel(os.path.join(PD, "key.xlsx"))
    key = key.dropna(subset=["Valence"])
    parts = [g.sample(min(len(g), CAP), random_state=0) for _, g in key.groupby("Context Category")]
    key = pd.concat(parts).reset_index(drop=True)
    wavs, meta = [], []
    adir = os.path.join(PD, "Soundwel Dataset - Audio and Spectrograms")
    for _, r in key.iterrows():
        p = os.path.join(adir, r["Audio Filename"])
        try:
            x, sr = sf.read(p, dtype="float32")
        except Exception:
            continue
        if x.ndim > 1:
            x = x.mean(1)
        if sr != SR:
            g = np.gcd(int(sr), SR)
            x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        x = x[: int(MAX_S * SR)]
        if len(x) < 1600:
            x = np.pad(x, (0, 1600 - len(x)))
        wavs.append(x)
        meta.append({"file": r["Audio Filename"], "team": r["Recording Team"],
                     "ctx": r["Context Category"], "pol": 1 if r["Valence"] == "Neg" else 0,
                     "sex": r["Sex"], "age": r["Age Category"],
                     "acoustic": [float(r[c]) if pd.notna(r[c]) else np.nan for c in ACOUSTIC]})
    return wavs, meta


def embed(wavs, model_id="microsoft/wavlm-base-plus"):
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
            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(wavs)}", flush=True)
    return np.stack(E, axis=1)


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


def loto(X, y, team):
    """leave-one-team-out; only teams with both classes are scored"""
    yp = np.full_like(y, -1)
    for t in np.unique(team):
        te = team == t
        if len(np.unique(y[te])) < 2:
            continue
        yp[te] = fit(X[~te], y[~te]).predict(X[te])
    m = yp >= 0
    return bacc(y[m], yp[m]), {t: bacc(y[team == t], yp[team == t]) for t in np.unique(team[m])}


def kfold(X, y, k=5):
    from sklearn.model_selection import StratifiedKFold
    yp = np.empty_like(y)
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=0).split(X, y):
        yp[te] = fit(X[tr], y[tr]).predict(X[te])
    return bacc(y, yp)


def main():
    cache_e = os.path.join(OUT, "emb_wavlm.npy"); cache_m = os.path.join(OUT, "meta.json")
    if os.path.exists(cache_e) and os.path.exists(cache_m):
        E = np.load(cache_e).astype(np.float32); meta = json.load(open(cache_m)); print("cached")
    else:
        wavs, meta = load_pigs()
        print(f"{len(wavs)} pig calls loaded; embedding...", flush=True)
        E = embed(wavs)
        np.save(cache_e, E.astype(np.float16)); json.dump(meta, open(cache_m, "w"))
    y = np.array([m["pol"] for m in meta]); team = np.array([m["team"] for m in meta])
    ctx = np.array([m["ctx"] for m in meta])
    A = np.array([m["acoustic"] for m in meta], dtype=np.float32)
    A = np.where(np.isnan(A), np.nanmean(A, 0), A)
    L = E.shape[0]
    print(f"\n{len(y)} calls · neg {int((y==1).sum())} pos {int((y==0).sum())} · teams {dict(zip(*np.unique(team, return_counts=True)))}")
    res = {}

    # ---------------- A. within-pig valence
    print("\n=== A. within-pig valence (chance 0.5) ===")
    print(f"{'layer':>5} {'LOTO':>6} {'random':>7} {'team-id':>8}")
    rows = []
    for li in range(L):
        Z = zgroup(E[li], np.zeros(len(y)))       # plain z-score
        lo, per = loto(Z, y, team)
        rd = kfold(Z, y)
        tid = float((kfold(Z, team) if False else 0))  # placeholder
        # team decodability: 6-way accuracy
        from sklearn.model_selection import StratifiedKFold
        yp = np.empty_like(team)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Z, team):
            yp[te] = fit(Z[tr], team[tr]).predict(Z[te])
        tid = float((yp == team).mean())
        rows.append({"layer": li, "loto": lo, "random": rd, "team_id": tid, "per_team": per})
        print(f"{li:>5} {lo:>6.3f} {rd:>7.3f} {tid:>8.3f}")
    best = max(rows, key=lambda r: r["loto"])
    print(f"  best LOTO layer {best['layer']}: {best['loto']:.3f}   per-team {{{', '.join(f'{k}:{v:.2f}' for k,v in best['per_team'].items())}}}")
    print(f"  mean over layers: LOTO {np.mean([r['loto'] for r in rows]):.3f}  random {np.mean([r['random'] for r in rows]):.3f}  team-id {np.mean([r['team_id'] for r in rows]):.3f} (chance {1/len(set(team)):.3f})")
    res["within"] = rows

    # published-feature baseline + duration alone
    Za = zgroup(A, np.zeros(len(y)))
    lo_a, per_a = loto(Za, y, team); rd_a = kfold(Za, y)
    dur = A[:, [ACOUSTIC.index("Dur")]]
    lo_d, _ = loto(zgroup(dur, np.zeros(len(y))), y, team)
    print(f"\n  paper's 18 acoustic features : LOTO {lo_a:.3f}   random {rd_a:.3f}")
    print(f"  duration alone               : LOTO {lo_d:.3f}")
    res["acoustic_baseline"] = {"loto": lo_a, "random": rd_a, "per_team": per_a, "duration_loto": lo_d}

    # ---------------- B. 3x3 transfer
    print("\n=== B. cross-species transfer, 3x3 (mean over layers, chance 0.5) ===")
    import cross_species as cs
    rows_cd = cs.load_all()
    Ecd = np.load(os.path.join(D, "out_cross", "emb_wavlm.npy")).astype(np.float32)
    sp_cd = np.array([r["species"] for r in rows_cd]); y_cd = np.array([r["pol"] for r in rows_cd])
    ind_cd = np.array([r["indiv"] for r in rows_cd])
    assert Ecd.shape[0] == L
    species = {"cat": (Ecd[:, sp_cd == "cat"], y_cd[sp_cd == "cat"], ind_cd[sp_cd == "cat"]),
               "dog": (Ecd[:, sp_cd == "dog"], y_cd[sp_cd == "dog"], ind_cd[sp_cd == "dog"]),
               "pig": (E, y, team)}
    names = ["cat", "dog", "pig"]
    M = np.zeros((3, 3)); Mp = np.zeros((3, 3))
    rng = np.random.default_rng(9)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            Ea, ya, ga = species[a]; Eb, yb, gb = species[b]
            accs = []
            for li in range(L):
                Za = zgroup(Ea[li], ga if a == "pig" else np.zeros(len(ya)))
                Zb = zgroup(Eb[li], gb if b == "pig" else np.zeros(len(yb)))
                if a == b:
                    if a == "pig":
                        accs.append(loto(Za, ya, ga)[0])
                    else:
                        # leave-one-individual-out
                        yp = np.empty_like(ya)
                        for g in np.unique(ga):
                            te = ga == g
                            yp[te] = fit(Za[~te], ya[~te]).predict(Za[te]) if len(np.unique(ya[~te])) > 1 else 0
                        accs.append(bacc(ya, yp))
                else:
                    accs.append(bacc(yb, fit(Za, ya).predict(Zb)))
            M[i, j] = np.mean(accs)
            # permutation null for off-diagonal: shuffle source labels within source group, 200 draws
            if a != b:
                null = []
                for _ in range(200):
                    yp_ = ya.copy()
                    for g in np.unique(ga):
                        m = ga == g; yp_[m] = rng.permutation(ya[m])
                    acc_ = []
                    for li in range(L):
                        Za = zgroup(Ea[li], ga if a == "pig" else np.zeros(len(ya)))
                        Zb = zgroup(Eb[li], gb if b == "pig" else np.zeros(len(yb)))
                        acc_.append(bacc(yb, fit(Za, yp_).predict(Zb)))
                    null.append(np.mean(acc_))
                Mp[i, j] = (np.sum(np.array(null) >= M[i, j]) + 1) / 201
            print(f"  {a:>3} -> {b:<3}  {M[i,j]:.3f}" + (f"   p={Mp[i,j]:.3f}" if a != b else "   (within)"), flush=True)
    res["matrix"] = {"names": names, "acc": M.tolist(), "p": Mp.tolist()}

    # placebo: pig sex -> cat affect
    sex = np.array([{"male": 1, "female": 0}.get(str(m["sex"]).lower(), -1) for m in meta])
    ok = sex >= 0
    accs = []
    for li in range(L):
        Zp = zgroup(E[li][ok], team[ok]); Zc = zgroup(species["cat"][0][li], np.zeros(len(species["cat"][1])))
        accs.append(bacc(species["cat"][1], fit(Zp, sex[ok]).predict(Zc)))
    print(f"  placebo pig SEX -> cat AFFECT: {np.mean(accs):.3f}")
    res["placebo_pigsex_to_cat"] = float(np.mean(accs))

    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print("\nwrote", os.path.join(OUT, "results.json"))


if __name__ == "__main__":
    main()
