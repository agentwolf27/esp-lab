"""Sanity-check + benchmark multiband_audio's MultibandTransform at 250 kHz."""
import torch
torch.set_num_threads(2)
import numpy as np, time, os, soundfile as sf, pandas as pd
import multiband_audio as mba

SR = 250_000
t = mba.MultibandTransform(sample_rate=SR, target_sr=16_000, max_freq=32_000)
print("bands:", t.get_band_info(), "n=", t.get_num_bands())

# --- synthetic tone test: where does a pure tone at f land in each band? ---
dur = 0.5
tt = np.arange(int(dur * SR)) / SR
for f0 in [2_000, 12_000, 20_000, 28_000, 40_000]:
    x = torch.from_numpy(np.sin(2 * np.pi * f0 * tt).astype(np.float32)).unsqueeze(0)
    b = t(x)                       # (1, 4, T')
    b = b[0].numpy()
    rms = np.sqrt((b ** 2).mean(1))
    # dominant freq in each band
    peaks = []
    for k in range(b.shape[0]):
        S = np.abs(np.fft.rfft(b[k] * np.hanning(len(b[k]))))
        peaks.append(np.fft.rfftfreq(len(b[k]), 1 / 16000)[S.argmax()])
    print(f"tone {f0:6d} Hz -> band rms {np.array2string(rms, precision=4)}  "
          f"peak-in-band(Hz) {[round(p) for p in peaks]}")

# --- benchmark on real bat calls ---
D = os.path.dirname(os.path.abspath(__file__))
sub = pd.read_csv(os.path.join(D, "bats", "subset2000.csv"))
files = sub["File Name"].tolist()[:5]
tot = 0.0
for f in files:
    p = os.path.join(D, "bats", "audio", f)
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    x = x[: int(6.0 * SR)]
    w = torch.from_numpy(x).unsqueeze(0)
    t0 = time.time()
    b = t(w)
    el = time.time() - t0
    tot += el
    print(f"  {f}  {len(x)/SR:.2f}s native -> bands {tuple(b.shape)}  {el:.2f}s")
print(f"mean {tot/len(files):.2f} s/file  -> 1000 files ~ {tot/len(files)*1000/60:.1f} min")
