"""
Per-clip acoustic feature extraction for the allometric-normalisation test.

Everything is analysed on a COMMON footing so that cross-species comparisons
are not artefacts of recording format:
  * mono, resampled to 16 kHz for F0 (cats are natively 8 kHz, so nothing
    above 4 kHz is real for them -- spectral features are therefore computed
    over the 0-4 kHz band only, for every species)
  * LPC / formant analysis is done at 8 kHz (0-4 kHz), the lowest common
    bandwidth in the three corpora
  * the unit of analysis is the FOCAL CALL, not the file. Cat files are single
    meows (~1.8 s), pig files are single calls (~0.3 s), but dog files are bout
    recordings (median 10 s, max 100 s). Comparing "duration" across those
    would compare segmentation conventions, not animals. We therefore locate
    the loudest contiguous energetic segment in each file and analyse that.
"""
from __future__ import annotations
import glob, json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(D, "out_allometry")
SR = 16_000
SR_LPC = 8_000
MAXSEG = 4.0          # cap on focal-segment length (s)

# ---------------------------------------------------------------- F0 ranges
# cat meow F0 is typically 350-800 Hz (Nicastro 2004; Schoetz 2019) -> 180-1500
# dog bark F0 typically 250-900 Hz, growls far lower (Yin & McCowan 2004) -> 110-2000
# pig calls span low grunts (~20-100 Hz) to screams (>1000 Hz); the Soundwel
# key's own F0Mean has median 59 Hz and 1st pct ~30 Hz, so 100 Hz would be far
# too high a floor -> 40-1500
F0RANGE = {"cat": (180.0, 1500.0), "dog": (110.0, 2000.0), "pig": (40.0, 1500.0)}


def _frame_len(fmin):
    # pyin needs >= ~4 periods of fmin inside a frame
    n = 4.0 * SR / fmin
    return int(2 ** np.ceil(np.log2(max(n, 512))))


def read_mono(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    x = np.ascontiguousarray(x, dtype=np.float64)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, sr // g)
    return x.astype(np.float64), sr


def focal_segment(x):
    """Loudest contiguous energetic run. Returns (i0, i1) sample indices."""
    import librosa
    fl, hl = 512, 128
    if len(x) < fl * 2:
        return 0, len(x)
    rms = librosa.feature.rms(y=x, frame_length=fl, hop_length=hl, center=True)[0]
    if rms.max() <= 0:
        return 0, len(x)
    act = rms > max(0.12 * rms.max(), 1e-6)
    # merge gaps <= 60 ms (~8 hops)
    gap = 8
    idx = np.flatnonzero(act)
    if idx.size == 0:
        return 0, len(x)
    runs, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > gap:
            runs.append((s, p)); s = i
        p = i
    runs.append((s, p))
    e2 = rms ** 2
    runs = [r for r in runs if (r[1] - r[0] + 1) * hl / SR >= 0.025] or runs
    best = max(runs, key=lambda r: e2[r[0]:r[1] + 1].sum())
    i0 = max(0, best[0] * hl - fl // 2)
    i1 = min(len(x), (best[1] + 1) * hl + fl // 2)
    if (i1 - i0) / SR > MAXSEG:                    # keep the loudest MAXSEG window
        w = int(MAXSEG * SR)
        c = e2[best[0]:best[1] + 1]
        k = int(np.argmax(np.convolve(c, np.ones(max(1, w // hl)), "valid"))) if len(c) * hl > w else 0
        i0 = max(0, (best[0] + k) * hl); i1 = min(len(x), i0 + w)
    return int(i0), int(i1)


def formant_dispersion(seg, order=12):
    """LPC formant dispersion (Fitch 1997; slope method of Reby & McComb 2003)
    over the 0-4 kHz band, which is all the cat corpus can offer."""
    from scipy.signal import resample_poly
    import librosa
    g = np.gcd(SR, SR_LPC)
    s = resample_poly(seg, SR_LPC // g, SR // g)
    if len(s) < 512:
        return np.nan, 0
    s = s - s.mean()
    if s.std() < 1e-9:
        return np.nan, 0
    s = librosa.effects.preemphasis(s, coef=0.97)
    # average the autocorrelation over frames for a stable estimate
    fl, hl = 512, 256
    F = librosa.util.frame(s, frame_length=fl, hop_length=hl) if len(s) >= fl else s[:, None]
    w = np.hanning(F.shape[0])[:, None]
    Fw = F * w
    ac = np.zeros(order + 1)
    for j in range(Fw.shape[1]):
        c = np.correlate(Fw[:, j], Fw[:, j], "full")[fl - 1:fl + order] if Fw.shape[0] == fl else None
        if c is None or len(c) < order + 1:
            continue
        ac += c
    if ac[0] <= 0:
        return np.nan, 0
    ac[0] *= 1.0001
    try:
        from scipy.linalg import solve_toeplitz
        a = solve_toeplitz((ac[:order], ac[:order]), ac[1:order + 1])
    except Exception:
        return np.nan, 0
    poly = np.concatenate([[1.0], -a])
    r = np.roots(poly)
    r = r[np.imag(r) > 0]
    if r.size == 0:
        return np.nan, 0
    f = np.angle(r) * SR_LPC / (2 * np.pi)
    bw = -0.5 * (SR_LPC / (2 * np.pi)) * np.log(np.abs(r))
    keep = (f > 150) & (f < 3800) & (bw < 1000)
    f = np.sort(f[keep])
    if f.size < 2:
        return np.nan, int(f.size)
    i = np.arange(1, f.size + 1) - 0.5           # Fi = Df * (i - 0.5)
    Df = float(np.dot(i, f) / np.dot(i, i))
    return Df, int(f.size)


def clip_features(path, species):
    import librosa
    x, sr_native = read_mono(path)
    if len(x) < 800:
        x = np.pad(x, (0, 800 - len(x)))
    i0, i1 = focal_segment(x)
    seg = x[i0:i1]
    if len(seg) < 400:
        seg = x
        i0, i1 = 0, len(x)
    fmin, fmax = F0RANGE[species]
    fl = _frame_len(fmin)
    hop = fl // 4
    pad = seg if len(seg) >= fl else np.pad(seg, (0, fl - len(seg)))
    try:
        f0, vflag, vprob = librosa.pyin(pad, fmin=fmin, fmax=fmax, sr=SR,
                                        frame_length=fl, hop_length=hop,
                                        fill_na=np.nan, resolution=0.15)
    except Exception:
        f0 = np.array([np.nan]); vflag = np.array([False]); vprob = np.array([0.0])
    v = np.isfinite(f0)
    nv = int(v.sum())
    f0v = f0[v]
    out = {
        "path": os.path.basename(path), "species": species, "sr_native": int(sr_native),
        "dur_file": len(x) / SR, "dur_call": len(seg) / SR,
        "n_f0_frames": int(len(f0)), "n_voiced": nv,
        "voiced_frac": float(nv / max(1, len(f0))),
        "f0_med": float(np.median(f0v)) if nv else np.nan,
        "f0_mean": float(np.mean(f0v)) if nv else np.nan,
        "f0_sd": float(np.std(f0v)) if nv > 1 else np.nan,
        "f0_p10": float(np.percentile(f0v, 10)) if nv else np.nan,
        "f0_p90": float(np.percentile(f0v, 90)) if nv else np.nan,
        "vprob_mean": float(np.nanmean(vprob)) if len(vprob) else np.nan,
    }
    # energy
    rms_seg = float(np.sqrt(np.mean(seg ** 2)) + 1e-12)
    bg = np.concatenate([x[:i0], x[i1:]]) if (i0 > 0 or i1 < len(x)) else np.array([0.0])
    rms_bg = float(np.sqrt(np.mean(bg ** 2)) + 1e-12) if bg.size > 10 else np.nan
    out["log_rms"] = float(np.log10(rms_seg))
    out["snr_db"] = float(20 * np.log10(rms_seg / rms_bg)) if np.isfinite(rms_bg) else np.nan
    # spectral shape restricted to 0-4 kHz (common band)
    n_fft = 1024
    S = np.abs(librosa.stft(seg if len(seg) >= n_fft else np.pad(seg, (0, n_fft - len(seg))),
                            n_fft=n_fft, hop_length=n_fft // 4))
    fr = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    m = fr <= 4000
    Sm = S[m]; frm = fr[m]
    w = Sm.sum(0) + 1e-12
    cen = (Sm * frm[:, None]).sum(0) / w
    out["centroid"] = float(np.mean(cen))
    out["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(
        seg, frame_length=512, hop_length=128)[0]))
    # DIAGNOSTIC ONLY: reported at two LPC orders to expose the order-dependence.
    Df, nf = formant_dispersion(seg, order=12)
    out["formant_disp"], out["n_formants"] = Df, nf
    Df8, nf8 = formant_dispersion(seg, order=8)
    out["formant_disp_o8"], out["n_formants_o8"] = Df8, nf8
    return out


# ------------------------------------------------------------------ corpora
def manifest():
    import pandas as pd
    rows = []
    for f in sorted(glob.glob(os.path.join(D, "catmeows", "**", "*.wav"), recursive=True)):
        p = os.path.basename(f)[:-4].split("_")
        rows.append(dict(path=f, species="cat", ctx=p[0], indiv="cat_" + p[1],
                         group="cat_" + p[1], breed=p[2], sex=p[3],
                         pol={"I": 1, "B": 0}.get(p[0], np.nan)))
    ann = pd.read_csv(os.path.join(D, "dogs", "annotations.csv"))
    for _, r in ann.iterrows():
        rows.append(dict(path=os.path.join(D, "dogs", "audio", r["filename"]),
                         species="dog", ctx=r["context"], indiv="dog_" + r["name"],
                         group="dog_" + r["name"], breed=r["breed"], sex=r["sex"],
                         mass_kg=float(r["weight"]), age_y=float(r["age"]),
                         pol={"aggression": 1, "play": 0}.get(r["context"], np.nan)))
    key = pd.read_excel(os.path.join(D, "pigs", "key.xlsx")).dropna(subset=["Valence"])
    CAP = int(os.environ.get("PIGCAP", "120"))
    parts = [g.sample(min(len(g), CAP), random_state=0)
             for _, g in key.groupby(["Recording Team", "Context Category"])]
    key = pd.concat(parts).reset_index(drop=True)
    adir = os.path.join(D, "pigs", "Soundwel Dataset - Audio and Spectrograms")
    for _, r in key.iterrows():
        rows.append(dict(path=os.path.join(adir, r["Audio Filename"]), species="pig",
                         ctx=r["Context Category"], indiv="pig_" + str(r["Recording Team"]),
                         group="pig_" + str(r["Recording Team"]), team=str(r["Recording Team"]),
                         age_cat=str(r["Age Category"]), sex=str(r["Sex"]),
                         pol={"Neg": 1, "Pos": 0}[r["Valence"]],
                         key_f0mean=float(r["F0Mean"]) if np.isfinite(r["F0Mean"]) else np.nan,
                         key_dur=float(r["Dur"]) if np.isfinite(r["Dur"]) else np.nan))
    return rows


def _work(args):
    i, r = args
    try:
        f = clip_features(r["path"], r["species"])
    except Exception as e:                                        # noqa: BLE001
        return i, {"path": os.path.basename(r["path"]), "species": r["species"],
                   "error": f"{type(e).__name__}: {e}"}
    return i, f


def main():
    import pandas as pd
    from multiprocessing import Pool
    rows = manifest()
    print(f"{len(rows)} clips: " + ", ".join(
        f"{s}={sum(1 for r in rows if r['species']==s)}" for s in ["cat", "dog", "pig"]), flush=True)
    res = [None] * len(rows)
    nw = int(os.environ.get("NWORK", "2"))
    with Pool(nw) as pool:
        for n, (i, f) in enumerate(pool.imap_unordered(_work, list(enumerate(rows)), chunksize=8)):
            res[i] = f
            if (n + 1) % 100 == 0:
                print(f"  {n+1}/{len(rows)}", flush=True)
    df = pd.DataFrame([{**{k: v for k, v in r.items() if k != "path"}, **f}
                       for r, f in zip(rows, res)])
    df.to_csv(os.path.join(OUT, "features.csv"), index=False)
    print("wrote", os.path.join(OUT, "features.csv"), df.shape, flush=True)
    for s in ["cat", "dog", "pig"]:
        d = df[df.species == s]
        print(f"  {s}: n={len(d)} f0_ok={np.isfinite(d.f0_med).mean():.3f} "
              f"f0_med={np.nanmedian(d.f0_med):.1f} Df_ok={np.isfinite(d.formant_disp).mean():.3f} "
              f"Df_med={np.nanmedian(d.formant_disp):.0f}", flush=True)


if __name__ == "__main__":
    main()
