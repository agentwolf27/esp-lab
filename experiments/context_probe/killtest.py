"""
E1 — the kill-test battery. Does dog->cat transfer survive?

The single most dangerous alternative explanation: CALL DURATION.
log-duration is the ONE handcrafted feature that separates the classes in the
SAME direction in both species (cat +0.28 SD, dog +0.31 SD). A same-sign
feature of d~0.3 predicts transfer accuracy around Phi(d/2) ~= 0.56 -- which is
exactly the band we observed (0.530-0.583). Mean-pooled SSL states encode clip
duration well. So "the probe found duration" is a live and sufficient null.

Tests, in order of lethality:
  1. duration-only transfer (1 feature). If this alone hits ~0.56, the
     embedding result explains nothing extra.
  2. each of the 4 handcrafted features alone -- shows WHY the 4-feature probe
     failed (it loads on energy, which anti-transfers), correcting the claim
     that "handcrafted features cannot transfer".
  3. duration-partialled embeddings: regress log-duration out of every
     embedding dimension, rerun the selection-free transfer statistic.
  4. ANIMAL-LEVEL permutation: permute context labels WITHIN each source
     animal. Clip-level permutation treats 308 correlated barks as
     independent and is anti-conservative when identity is 79% decodable.
  5. direction-asymmetry test: is dog->cat actually different from cat->dog,
     or did we over-read two noisy numbers?
  6. individual x class contingency tables (never printed before).
"""
from __future__ import annotations

import json, os, warnings
import numpy as np

warnings.filterwarnings("ignore")
D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "out_cross")
N_PERM = 2000


def fit_predict(Xa, ya, Xb):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(Xa, ya).predict(Xb)


def bacc(a, b):
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(a, b)


def zw(X, sp):
    Z = X.copy()
    for s in np.unique(sp):
        m = sp == s
        Z[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-8)
    return Z


def perm_within_animal(y, ind, rng):
    """Permute labels WITHIN each animal: preserves animal structure, kills context."""
    yp = y.copy()
    for g in np.unique(ind):
        m = ind == g
        yp[m] = rng.permutation(y[m])
    return yp


def main():
    import cross_species as cs
    from room_control import background_vec  # noqa
    from validate_transfer import acoustic_features

    rows = cs.load_all()
    sp = np.array([r["species"] for r in rows])
    y = np.array([r["pol"] for r in rows])
    ind = np.array([r["indiv"] for r in rows])
    c, d = sp == "cat", sp == "dog"
    E = np.load(os.path.join(OUT, "emb_wavlm.npy")).astype(np.float32)
    A = acoustic_features(rows)                     # [log_dur, log_energy, centroid, zcr]
    names = ["log_dur", "log_energy", "centroid", "zcr"]
    rng = np.random.default_rng(23)
    out = {}

    # ---------------------------------------------------------- 6. contingency
    print("=== individual x class (never shown before) ===")
    for s, m in [("cat", c), ("dog", d)]:
        print(f"  {s}:")
        for g in np.unique(ind[m]):
            mm = ind == g
            print(f"    {g:<13s} pos={int((y[mm]==0).sum()):3d}  neg={int((y[mm]==1).sum()):3d}")
    out["contingency"] = {g: {"pos": int((y[ind==g]==0).sum()), "neg": int((y[ind==g]==1).sum())}
                          for g in np.unique(ind)}

    # ---------------------------------------------------------- 1&2. single features
    print("\n=== single-feature transfer (THE decisive test) ===")
    single = {}
    for j, nm in enumerate(names):
        Za = zw(A[:, [j]], sp)
        cd = bacc(y[d], fit_predict(Za[c], y[c], Za[d]))
        dc = bacc(y[c], fit_predict(Za[d], y[d], Za[c]))
        single[nm] = {"cat_to_dog": cd, "dog_to_cat": dc}
        flag = "  <-- SAME-SIGN, the danger" if nm == "log_dur" else ""
        print(f"  {nm:<11s} cat->dog {cd:.3f}   dog->cat {dc:.3f}{flag}")
    Za4 = zw(A, sp)
    print(f"  {'all 4':<11s} cat->dog {bacc(y[d], fit_predict(Za4[c], y[c], Za4[d])):.3f}   "
          f"dog->cat {bacc(y[c], fit_predict(Za4[d], y[d], Za4[c])):.3f}")
    out["single_feature"] = single

    # ---------------------------------------------------------- 3. duration-partialled embeddings
    print("\n=== duration-partialled embeddings (regress log-dur out, per species) ===")
    dur = A[:, 0:1]
    raw_dc, par_dc, raw_cd, par_cd = [], [], [], []
    for li in range(E.shape[0]):
        X = E[li].copy()
        Xp = X.copy()
        for s in (True, False):
            m = c if s else d
            Dm = np.c_[np.ones(m.sum()), (dur[m] - dur[m].mean()) / (dur[m].std() + 1e-8)]
            beta, *_ = np.linalg.lstsq(Dm, X[m], rcond=None)
            Xp[m] = X[m] - Dm @ beta                 # residual after removing duration
        Z, Zp = zw(X, sp), zw(Xp, sp)
        raw_dc.append(bacc(y[c], fit_predict(Z[d], y[d], Z[c])))
        par_dc.append(bacc(y[c], fit_predict(Zp[d], y[d], Zp[c])))
        raw_cd.append(bacc(y[d], fit_predict(Z[c], y[c], Z[d])))
        par_cd.append(bacc(y[d], fit_predict(Zp[c], y[c], Zp[d])))
    print(f"  dog->cat  raw {np.mean(raw_dc):.3f}  ->  duration-partialled {np.mean(par_dc):.3f}")
    print(f"  cat->dog  raw {np.mean(raw_cd):.3f}  ->  duration-partialled {np.mean(par_cd):.3f}")
    out["partialled"] = {"dog_to_cat_raw": float(np.mean(raw_dc)),
                         "dog_to_cat_partialled": float(np.mean(par_dc)),
                         "cat_to_dog_raw": float(np.mean(raw_cd)),
                         "cat_to_dog_partialled": float(np.mean(par_cd)),
                         "per_layer_partialled_dc": [float(v) for v in par_dc]}

    # ---------------------------------------------------------- 4. animal-level permutation
    print(f"\n=== animal-level permutation null ({N_PERM} draws, labels shuffled WITHIN animal) ===")
    Zs = [zw(E[li], sp) for li in range(E.shape[0])]
    obs_dc = float(np.mean([bacc(y[c], fit_predict(Z[d], y[d], Z[c])) for Z in Zs]))
    obs_cd = float(np.mean([bacc(y[d], fit_predict(Z[c], y[c], Z[d])) for Z in Zs]))
    null_dc, null_cd = [], []
    for i in range(N_PERM):
        ypd = perm_within_animal(y, ind, rng)
        null_dc.append(np.mean([bacc(y[c], fit_predict(Z[d], ypd[d], Z[c])) for Z in Zs]))
        null_cd.append(np.mean([bacc(y[d], fit_predict(Z[c], ypd[c], Z[d])) for Z in Zs]))
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{N_PERM}")
    null_dc, null_cd = np.array(null_dc), np.array(null_cd)
    p_dc = float(((null_dc >= obs_dc).sum() + 1) / (N_PERM + 1))
    p_cd = float(((null_cd >= obs_cd).sum() + 1) / (N_PERM + 1))
    print(f"  dog->cat {obs_dc:.3f}  null {null_dc.mean():.3f}±{null_dc.std():.3f}  p = {p_dc:.4f}")
    print(f"  cat->dog {obs_cd:.3f}  null {null_cd.mean():.3f}±{null_cd.std():.3f}  p = {p_cd:.4f}")
    out["animal_perm"] = {"obs_dc": obs_dc, "obs_cd": obs_cd, "p_dc": p_dc, "p_cd": p_cd,
                          "null_dc_mean": float(null_dc.mean()), "null_dc_sd": float(null_dc.std())}

    # ---------------------------------------------------------- 5. asymmetry
    diff = obs_dc - obs_cd
    null_diff = null_dc - null_cd
    p_diff = float(((np.abs(null_diff) >= abs(diff)).sum() + 1) / (N_PERM + 1))
    print(f"\n=== is the asymmetry real? ===")
    print(f"  dog->cat minus cat->dog = {diff:+.3f}   null |diff| p = {p_diff:.4f}")
    print("  -> " + ("asymmetry is significant" if p_diff < .05 else
                     "NOT significant: report 'cat->dog underpowered', never 'cat->dog fails'"))
    out["asymmetry"] = {"diff": float(diff), "p": p_diff}

    json.dump(out, open(os.path.join(OUT, "killtest.json"), "w"), indent=1)

    # ---------------------------------------------------------- verdict
    print("\n" + "=" * 66)
    print("VERDICT")
    print("=" * 66)
    dur_dc = single["log_dur"]["dog_to_cat"]
    drop = np.mean(raw_dc) - np.mean(par_dc)
    if dur_dc >= obs_dc - 0.02:
        print(f"  ** DURATION EXPLAINS IT ** duration alone transfers at {dur_dc:.3f}, ")
        print(f"     vs {obs_dc:.3f} for the full embedding. The embedding adds nothing.")
    elif np.mean(par_dc) < 0.52:
        print(f"  ** DURATION EXPLAINS IT ** partialling duration out collapses")
        print(f"     dog->cat from {np.mean(raw_dc):.3f} to {np.mean(par_dc):.3f}.")
    elif p_dc < 0.05:
        print(f"  SURVIVES: duration alone gives {dur_dc:.3f}; partialling duration out")
        print(f"  leaves {np.mean(par_dc):.3f} (drop of {drop:.3f}); animal-level p = {p_dc:.4f}.")
    else:
        print(f"  DOES NOT SURVIVE animal-level permutation (p = {p_dc:.4f}), even though")
        print(f"  duration alone only gives {dur_dc:.3f}. Underpowered, not disproven.")


if __name__ == "__main__":
    main()
