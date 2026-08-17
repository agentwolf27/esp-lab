"""
The confound that would kill the transfer result: ROOM / CHANNEL.

CatMeows induced isolation by MOVING THE CAT TO AN UNFAMILIAR ROOM. So the
negative class is, by construction, recorded in a different physical space
from the positive class. If dog aggression was likewise recorded in a
different setting from play, then a probe could learn "reverberant/noisy
room" instead of "distressed animal" -- and that WOULD transfer across
species, while having nothing to do with affect.

Test: throw the vocalization away and keep only the background.
For each clip take the quietest 30% of frames (the parts between/around the
call) and build features from those alone. Then run the identical pipeline.

  If background-only transfers as well as the full clip  -> it is the room.
  If background-only is at chance but full clip is not   -> it is the animal.

Also reported: a reverberation proxy (decay of the energy envelope after the
loudest frame) and the noise floor, per class per species.
"""
from __future__ import annotations

import json, os, warnings
import numpy as np
from scipy.signal import stft

warnings.filterwarnings("ignore")
D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "out_cross")
SR = 16_000
N_PERM = 300


def frames(x):
    f, _, Z = stft(x, fs=SR, nperseg=400, noverlap=240, nfft=512)
    return f, np.abs(Z) ** 2                     # (F, T)


def background_vec(x, q=0.30):
    """Log-mel-ish summary of the QUIETEST frames only."""
    f, P = frames(x)
    e = P.sum(0)
    k = max(3, int(len(e) * q))
    idx = np.argsort(e)[:k]                      # quietest frames
    B = P[:, idx]
    logB = np.log(B + 1e-12)
    # coarse 20-band summary + noise-floor stats
    bands = np.array_split(logB, 20, axis=0)
    v = [b.mean() for b in bands] + [b.std() for b in bands]
    v += [np.log(e[idx].mean() + 1e-12), np.log(e.max() + 1e-12) - np.log(e[idx].mean() + 1e-12)]
    return np.array(v, dtype=np.float32)


def reverb_proxy(x):
    """How fast energy decays after the peak: a crude RT60-ish measure."""
    _, P = frames(x)
    e = np.log(P.sum(0) + 1e-12)
    pk = int(np.argmax(e))
    tail = e[pk:]
    if len(tail) < 4:
        return 0.0
    n = min(len(tail), 25)
    t = np.arange(n)
    slope = np.polyfit(t, tail[:n], 1)[0]        # more negative = faster decay = drier
    return float(slope)


def fit_predict(Xa, ya, Xb):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(Xa, ya).predict(Xb)


def zwithin(X, sp):
    Z = X.copy()
    for s in np.unique(sp):
        m = sp == s
        Z[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-8)
    return Z


def main():
    from sklearn.metrics import balanced_accuracy_score as bacc
    import cross_species as cs

    rows = cs.load_all()
    sp = np.array([r["species"] for r in rows])
    y = np.array([r["pol"] for r in rows])
    ind = np.array([r["indiv"] for r in rows])
    c, d = sp == "cat", sp == "dog"
    print(f"{len(rows)} clips\n")

    print("building background-only features (quietest 30% of frames)...")
    B = np.stack([background_vec(r["wav"]) for r in rows])
    RV = np.array([reverb_proxy(r["wav"]) for r in rows])

    rng = np.random.default_rng(3)
    Zb = zwithin(B, sp)
    res = {}

    print("\n=== BACKGROUND-ONLY transfer (the room test) ===")
    cd = bacc(y[d], fit_predict(Zb[c], y[c], Zb[d]))
    dc = bacc(y[c], fit_predict(Zb[d], y[d], Zb[c]))
    ncd = np.array([bacc(y[d], fit_predict(Zb[c], rng.permutation(y[c]), Zb[d])) for _ in range(N_PERM)])
    ndc = np.array([bacc(y[c], fit_predict(Zb[d], rng.permutation(y[d]), Zb[c])) for _ in range(N_PERM)])
    p_cd = float((ncd >= cd).mean()); p_dc = float((ndc >= dc).mean())
    print(f"  cat->dog {cd:.3f} (p={p_cd:.3f})   dog->cat {dc:.3f} (p={p_dc:.3f})")
    print(f"  [compare: WavLM full-clip mean-over-layers dog->cat 0.583, p=0.0033]")
    res["background_transfer"] = {"cat_to_dog": cd, "dog_to_cat": dc, "p_cd": p_cd, "p_dc": p_dc}

    print("\n=== BACKGROUND-ONLY within species (can the room alone call the context?) ===")
    from sklearn.linear_model import LogisticRegression
    for s, m in [("cat", c), ("dog", d)]:
        yp = np.empty_like(y[m]); Xs, ys, gs = Zb[m], y[m], ind[m]
        for g in np.unique(gs):
            te = gs == g
            if len(np.unique(ys[~te])) < 2:
                yp[te] = 0; continue
            yp[te] = LogisticRegression(max_iter=3000, class_weight="balanced").fit(
                Xs[~te], ys[~te]).predict(Xs[te])
        a = bacc(ys, yp)
        print(f"  {s}: {a:.3f}   <- if this is high, the room predicts the label within species")
        res[f"background_within_{s}"] = a

    print("\n=== reverb proxy + noise floor by class (mean ± sd, in SD units) ===")
    for s, m in [("cat", c), ("dog", d)]:
        r = (RV[m] - RV[m].mean()) / (RV[m].std() + 1e-8)
        nf = B[m][:, -2]; nf = (nf - nf.mean()) / (nf.std() + 1e-8)
        print(f"  {s}: decay-slope  neg {r[y[m]==1].mean():+.2f}  pos {r[y[m]==0].mean():+.2f}   "
              f"| noise-floor  neg {nf[y[m]==1].mean():+.2f}  pos {nf[y[m]==0].mean():+.2f}")
        res[f"reverb_{s}"] = {"neg": float(r[y[m]==1].mean()), "pos": float(r[y[m]==0].mean()),
                              "nf_neg": float(nf[y[m]==1].mean()), "nf_pos": float(nf[y[m]==0].mean())}

    json.dump(res, open(os.path.join(OUT, "room_control.json"), "w"), indent=1)
    print(f"\nwrote {OUT}/room_control.json")

    print("\n=== VERDICT ===")
    if p_dc < 0.05 and dc > 0.57:
        print("  ROOM CONFOUND LIKELY: background alone transfers about as well as the")
        print("  full clip. The 'affect direction' may be channel/room, not animal.")
    elif p_dc < 0.05:
        print("  PARTIAL: background transfers weakly but significantly. Some channel")
        print("  information is shared; the embedding result needs a matched-room test.")
    else:
        print("  ROOM CONFOUND NOT SUPPORTED: background-only does not transfer, while")
        print("  the full-clip embedding does. Consistent with the signal being in the")
        print("  vocalization itself rather than the recording environment.")


if __name__ == "__main__":
    main()
