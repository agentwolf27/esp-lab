"""
Context probe, experiment 1 (baseline): can a call's *context* be predicted from its sound?

Dataset: CatMeows (Ludovico et al. 2021, CC-BY-4.0). 440 meows, 21 cats, 3 contexts:
    B = brushing (at home, with owner)
    F = waiting for food
    I = isolation in an unfamiliar environment
Filename encodes everything:  <ctx>_<catid>_<breed>_<sex>_<owner>_<session>_<n>.wav

Embedding: 40-dim log-mel spectrogram, summarised per clip into a 96-dim vector
(mean + std of 20 MFCCs, mean + std of log-mel energy, F0-ish spectral centroid, duration).
No deep model. Deliberately: this is the floor. If a frozen encoder can't beat this,
it isn't earning its keep.

Splits: LEAVE-ONE-CAT-OUT. Never random. Every meow from a given cat is either all-train
or all-test. This is the single most important line in the file — random splits let the
classifier learn "which cat" instead of "which context", and cats are not equally
represented across contexts, so it would look great and mean nothing.

Runs on stock anaconda python (numpy, scipy, sklearn, matplotlib). No torch, no librosa.

    python context_probe.py <path-to-catmeows-dir> <out-dir>
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from scipy.fftpack import dct
from scipy.io import wavfile
from scipy.signal import resample_poly, stft

# ------------------------------------------------------------------ audio -> features
SR = 16_000
N_MELS = 40
N_MFCC = 20


def load_mono_16k(path: str) -> np.ndarray:
    sr, x = wavfile.read(path)
    if x.dtype.kind == "i":
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    elif x.dtype.kind == "u":
        x = (x.astype(np.float32) - 128) / 128.0
    else:
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def mel_filterbank(n_fft: int, sr: int, n_mels: int, fmin: float = 50.0, fmax: float | None = None):
    fmax = fmax or sr / 2
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz = mel2hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        lo, c, hi = bins[i], bins[i + 1], bins[i + 2]
        if c == lo:
            c += 1
        if hi == c:
            hi += 1
        fb[i, lo:c] = (np.arange(lo, c) - lo) / (c - lo)
        fb[i, c:hi] = (hi - np.arange(c, hi)) / (hi - c)
    return fb


_FB = mel_filterbank(512, SR, N_MELS)


def logmel(x: np.ndarray) -> np.ndarray:
    """(n_mels, T) log-mel spectrogram, 25 ms window / 10 ms hop."""
    _, _, Z = stft(x, fs=SR, nperseg=400, noverlap=240, nfft=512, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    M = _FB @ P
    return np.log(M + 1e-8)


def clip_vector(x: np.ndarray) -> np.ndarray:
    """One fixed-length vector per clip. This is our 'embedding' for the baseline."""
    L = logmel(x)                                   # (40, T)
    mf = dct(L, type=2, axis=0, norm="ortho")[:N_MFCC]  # (20, T)
    energy = L.mean(axis=0)                          # (T,)
    # spectral centroid over the mel axis, as a cheap pitch/brightness proxy
    w = np.exp(L - L.max(axis=0, keepdims=True))
    centroid = (w * np.arange(N_MELS)[:, None]).sum(0) / (w.sum(0) + 1e-8)
    feats = np.concatenate([
        mf.mean(1), mf.std(1),                       # 40
        L.mean(1),                                   # 40   mean log-mel per band
        [energy.mean(), energy.std(),
         centroid.mean(), centroid.std(),
         np.log(len(x) / SR + 1e-3)],               # 5
    ])
    return feats.astype(np.float32)                  # 85-dim


# ------------------------------------------------------------------ dataset
def parse(path: str):
    name = os.path.basename(path)[:-4]
    parts = name.split("_")
    return {"ctx": parts[0], "cat": parts[1], "breed": parts[2], "sex": parts[3],
            "owner": parts[4], "session": parts[5]}


CTX_NAME = {"B": "brushing", "F": "waiting for food", "I": "isolation"}


def build(cat_dir: str):
    files = sorted(glob.glob(os.path.join(cat_dir, "**", "*.wav"), recursive=True))
    X, meta = [], []
    for f in files:
        try:
            x = load_mono_16k(f)
        except Exception as e:                       # noqa: BLE001
            print("skip", f, e)
            continue
        if len(x) < 400:
            continue
        X.append(clip_vector(x))
        m = parse(f)
        m["dur"] = len(x) / SR
        meta.append(m)
    return np.stack(X), meta


# ------------------------------------------------------------------ probes
def leave_one_cat_out(X, y, cats, make_clf):
    from sklearn.preprocessing import StandardScaler
    y_pred = np.empty_like(y)
    for c in np.unique(cats):
        te = cats == c
        tr = ~te
        sc = StandardScaler().fit(X[tr])
        clf = make_clf().fit(sc.transform(X[tr]), y[tr])
        y_pred[te] = clf.predict(sc.transform(X[te]))
    return y_pred


def random_split_cv(X, y, make_clf, seed=0):
    """The WRONG way, kept only to show how much it inflates the number."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    y_pred = np.empty_like(y)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        y_pred[te] = make_clf().fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    return y_pred


def main(cat_dir: str, out: str) -> None:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
    from sklearn.neighbors import KNeighborsClassifier
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out, exist_ok=True)
    X, meta = build(cat_dir)
    y = np.array([m["ctx"] for m in meta])
    cats = np.array([m["cat"] for m in meta])
    labels = ["B", "F", "I"]
    print(f"{len(y)} meows · {len(set(cats))} cats · dims={X.shape[1]}")
    for l in labels:
        print(f"  {CTX_NAME[l]:<18s} {int((y == l).sum()):4d}  from {len(set(cats[y == l]))} cats")

    results = {}
    clfs = {
        "logreg": lambda: LogisticRegression(max_iter=2000, C=0.5),
        "knn5": lambda: KNeighborsClassifier(5),
    }
    print("\n=== leave-one-cat-out (honest) ===")
    for name, mk in clfs.items():
        yp = leave_one_cat_out(X, y, cats, mk)
        ba = balanced_accuracy_score(y, yp); f1 = f1_score(y, yp, average="macro")
        results[f"loco_{name}"] = {"bal_acc": ba, "macro_f1": f1}
        print(f"  {name:<8s} balanced acc {ba:.3f}   macro-F1 {f1:.3f}   (chance 0.333)")
        if name == "logreg":
            cm = confusion_matrix(y, yp, labels=labels)
    print("\n=== random 5-fold (leaky, for contrast) ===")
    for name, mk in clfs.items():
        yp = random_split_cv(X, y, mk)
        ba = balanced_accuracy_score(y, yp)
        results[f"random_{name}"] = {"bal_acc": ba}
        print(f"  {name:<8s} balanced acc {ba:.3f}")

    # how much of the signal is "which cat" rather than "which context"?
    print("\n=== identity leak check: predict the CAT from the same features ===")
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    ok = np.isin(cats, [c for c in np.unique(cats) if (cats == c).sum() >= 5])
    yp = np.empty(ok.sum(), dtype=object); Xi, ci = X[ok], cats[ok]
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xi, ci):
        sc = StandardScaler().fit(Xi[tr])
        yp[te] = LogisticRegression(max_iter=2000).fit(sc.transform(Xi[tr]), ci[tr]).predict(sc.transform(Xi[te]))
    id_acc = float((yp == ci).mean()); n_id = len(np.unique(ci))
    results["identity_acc"] = {"acc": id_acc, "n_cats": n_id, "chance": 1 / n_id}
    print(f"  cat identity acc {id_acc:.3f} over {n_id} cats (chance {1/n_id:.3f})")
    print("  -> the same features carry strong identity information; that is exactly")
    print("     why a random split over-reports context accuracy.")

    # ---------------------------------------------------------------- figures
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    P = PCA(2).fit(Xs); Z = P.transform(Xs)
    col = {"B": "#0A626D", "F": "#C87A12", "I": "#8A3459"}
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)
    for l in labels:
        m = y == l
        ax.scatter(Z[m, 0], Z[m, 1], s=18, alpha=.75, c=col[l], label=f"{CTX_NAME[l]} (n={m.sum()})", edgecolors="none")
    ax.set_xlabel(f"PC1 ({P.explained_variance_ratio_[0]*100:.0f}%)"); ax.set_ylabel(f"PC2 ({P.explained_variance_ratio_[1]*100:.0f}%)")
    ax.set_title("CatMeows · 85-d MFCC/log-mel clip vectors · PCA", fontsize=11)
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "pca_context.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=150)
    cmn = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([CTX_NAME[l] for l in labels], rotation=20, ha="right", fontsize=8)
    ax.set_yticklabels([CTX_NAME[l] for l in labels], fontsize=8)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i,j]:.2f}\n({cm[i,j]})", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > .5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("leave-one-cat-out · logistic regression", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, "confusion_loco.png")); plt.close(fig)

    # per-cat accuracy: does it work for every animal or just a few?
    yp = leave_one_cat_out(X, y, cats, clfs["logreg"])
    per_cat = {c: float((yp[cats == c] == y[cats == c]).mean()) for c in np.unique(cats)}
    results["per_cat_acc"] = per_cat
    fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=150)
    ks = sorted(per_cat, key=per_cat.get)
    ax.bar(range(len(ks)), [per_cat[k] for k in ks], color="#0A626D")
    ax.axhline(1/3, ls="--", c="#8A3459", lw=1, label="chance")
    ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks, rotation=90, fontsize=6)
    ax.set_ylabel("accuracy on held-out cat"); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False); ax.set_title("does it generalise to a cat it never heard?", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, "per_cat.png")); plt.close(fig)

    np.save(os.path.join(out, "X_mfcc.npy"), X)
    json.dump({"meta": meta, "results": results}, open(os.path.join(out, "results.json"), "w"), indent=1)
    print(f"\nwrote {out}/  (pca_context.png, confusion_loco.png, per_cat.png, X_mfcc.npy, results.json)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "out")
