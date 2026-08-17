"""
E4 -- specificity and the residual.

(1) PLACEBO AXIS. If ANY dog binary predicts ANY cat binary, then "transfer" is
    generic corpus alignment, not affect. Test three mismatched pairs:
        dog SEX     -> cat AFFECT     must be ~0.5
        dog AFFECT  -> cat SEX        must be ~0.5
        cat SEX     -> dog AFFECT     must be ~0.5
    and, for reference, the matched-but-non-affect pair:
        dog SEX     -> cat SEX        (a real vocal-sex axis may exist; report it)

(2) RESIDUAL P-VALUE. After regressing log-duration out of every embedding
    dimension, dog->cat was 0.552. Is that significant with animal-level
    permutation? 500 draws (each = 13 layers x 1 fit).

(3) AUC alongside balanced accuracy for the main dog->cat transfer -- AUC is
    immune to the intercept-transfer artifact that class_weight='balanced'
    could induce across corpora with inverted imbalance.
"""
from __future__ import annotations

import json, os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "out_cross")
N_PERM = 500


def fit(Xa, ya):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(Xa, ya)


def bacc(a, b):
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(a, b)


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)


def zw(X, sp):
    Z = X.copy()
    for s in np.unique(sp):
        m = sp == s
        Z[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-8)
    return Z


def perm_within(y, ind, rng):
    yp = y.copy()
    for g in np.unique(ind):
        m = ind == g
        yp[m] = rng.permutation(y[m])
    return yp


def main():
    import cross_species as cs
    from validate_transfer import acoustic_features

    rows = cs.load_all()
    sp = np.array([r["species"] for r in rows])
    y = np.array([r["pol"] for r in rows])
    ind = np.array([r["indiv"] for r in rows])
    c, d = sp == "cat", sp == "dog"
    E = np.load(os.path.join(OUT, "emb_wavlm.npy")).astype(np.float32)
    A = acoustic_features(rows)
    L = E.shape[0]
    rng = np.random.default_rng(5)
    out = {}

    # ---- sex labels
    ann = pd.read_csv(os.path.join(D, "dogs", "annotations.csv"))
    dog_sex_map = {r["filename"]: r["sex"] for _, r in ann.iterrows()}
    sex = np.array([
        (os.path.basename(r["path"]).split("_")[3][0] if r["species"] == "cat"
         else dog_sex_map.get(os.path.basename(r["path"]), "?"))
        for r in rows])
    # normalise: cat 'F'/'M' ; dog 'female'/'male'
    sex = np.array([{"F": 0, "M": 1, "female": 0, "male": 1}.get(s, -1) for s in sex])
    ok = sex >= 0
    print(f"sex labels: cat F={int(((sex==0)&c).sum())} M={int(((sex==1)&c).sum())} | "
          f"dog F={int(((sex==0)&d).sum())} M={int(((sex==1)&d).sum())}")

    # ---- (1) placebo axes, mean over layers
    print("\n=== placebo / specificity (mean over 13 layers, chance 0.5) ===")
    def transfer(src_mask, src_y, tgt_mask, tgt_y):
        accs = []
        for li in range(L):
            Z = zw(E[li], sp)
            m = fit(Z[src_mask], src_y)
            accs.append(bacc(tgt_y, m.predict(Z[tgt_mask])))
        return float(np.mean(accs))
    pairs = {
        "dog AFFECT -> cat AFFECT (the claim)": (d, y[d], c, y[c]),
        "dog SEX    -> cat AFFECT (placebo)":   (d & ok, sex[d & ok], c, y[c]),
        "dog AFFECT -> cat SEX    (placebo)":   (d, y[d], c & ok, sex[c & ok]),
        "cat SEX    -> dog AFFECT (placebo)":   (c & ok, sex[c & ok], d, y[d]),
        "dog SEX    -> cat SEX    (matched, non-affect)": (d & ok, sex[d & ok], c & ok, sex[c & ok]),
    }
    for name, (sm, sy, tm, ty) in pairs.items():
        v = transfer(sm, sy, tm, ty)
        out[name] = v
        print(f"  {name:<46s} {v:.3f}")

    # ---- (2) residual permutation p (duration partialled)
    print(f"\n=== duration-partialled dog->cat, animal-level permutation ({N_PERM} draws) ===")
    dur = A[:, 0:1]
    Zp = []
    for li in range(L):
        X = E[li].copy(); Xp = X.copy()
        for m in (c, d):
            Dm = np.c_[np.ones(m.sum()), (dur[m] - dur[m].mean()) / (dur[m].std() + 1e-8)]
            beta, *_ = np.linalg.lstsq(Dm, X[m], rcond=None)
            Xp[m] = X[m] - Dm @ beta
        Zp.append(zw(Xp, sp))
    obs = float(np.mean([bacc(y[c], fit(Z[d], y[d]).predict(Z[c])) for Z in Zp]))
    null = []
    for i in range(N_PERM):
        yp = perm_within(y, ind, rng)
        null.append(np.mean([bacc(y[c], fit(Z[d], yp[d]).predict(Z[c])) for Z in Zp]))
    null = np.array(null)
    p = float(((null >= obs).sum() + 1) / (N_PERM + 1))
    print(f"  residual dog->cat {obs:.3f}   null {null.mean():.3f}±{null.std():.3f}   p = {p:.4f}")
    out["residual"] = {"obs": obs, "null_mean": float(null.mean()), "null_sd": float(null.std()), "p": p}

    # ---- (3) AUC for main transfer
    print("\n=== AUC (intercept-free) for dog->cat, per layer ===")
    aucs, aucs_res = [], []
    for li in range(L):
        Z = zw(E[li], sp)
        m = fit(Z[d], y[d]); aucs.append(auc(y[c], m.decision_function(Z[c])))
        m2 = fit(Zp[li][d], y[d]); aucs_res.append(auc(y[c], m2.decision_function(Zp[li][c])))
    print(f"  raw      : mean {np.mean(aucs):.3f}   per-layer {' '.join(f'{v:.2f}' for v in aucs)}")
    print(f"  residual : mean {np.mean(aucs_res):.3f}   per-layer {' '.join(f'{v:.2f}' for v in aucs_res)}")
    out["auc"] = {"raw_mean": float(np.mean(aucs)), "residual_mean": float(np.mean(aucs_res)),
                  "raw": [float(v) for v in aucs], "residual": [float(v) for v in aucs_res]}

    json.dump(out, open(os.path.join(OUT, "placebo.json"), "w"), indent=1)
    print("\n=== VERDICT ===")
    plc = [out[k] for k in out if "placebo" in k]
    if max(plc) < 0.53:
        print(f"  placebos at chance (max {max(plc):.3f}): the transfer is SPECIFIC to the affect axis.")
    else:
        print(f"  a placebo transferred at {max(plc):.3f}: generic corpus alignment is in play. Danger.")
    print(f"  residual after duration: {obs:.3f}, p={p:.4f} -> "
          + ("significant" if p < .05 else "NOT significant; the non-duration part is not established"))


if __name__ == "__main__":
    main()
