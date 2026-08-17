"""Characterise multiband_audio heterodyne: sideband folding + real bat band energy."""
import torch
torch.set_num_threads(2)
import numpy as np, os, soundfile as sf, pandas as pd
import multiband_audio as mba

SR = 250_000
t = mba.MultibandTransform(sample_rate=SR, target_sr=16_000, max_freq=32_000)
dur = 0.5
tt = np.arange(int(dur * SR)) / SR

print("=== tone response, band 1 = [8000,16000], centre 12000 ===")
print(f"{'tone':>7} {'phase':>5} | {'rms b0':>8} {'rms b1':>8} {'rms b2':>8} {'rms b3':>8} | peak b1")
for f0 in [9_000, 10_000, 12_000, 14_000, 15_000]:
    for ph_name, ph in [("sin", 0.0), ("cos", np.pi / 2)]:
        x = torch.from_numpy(np.sin(2 * np.pi * f0 * tt + ph).astype(np.float32)).unsqueeze(0)
        b = t(x)[0].numpy()
        rms = np.sqrt((b ** 2).mean(1))
        S = np.abs(np.fft.rfft(b[1] * np.hanning(b.shape[1])))
        pk = np.fft.rfftfreq(b.shape[1], 1 / 16000)[S.argmax()]
        print(f"{f0:7d} {ph_name:>5} | {rms[0]:8.4f} {rms[1]:8.4f} {rms[2]:8.4f} {rms[3]:8.4f} | {pk:6.0f} Hz")

print("\n=== reference: tone in band 0 ===")
for f0 in [1_000, 4_000, 7_000]:
    x = torch.from_numpy(np.sin(2 * np.pi * f0 * tt).astype(np.float32)).unsqueeze(0)
    b = t(x)[0].numpy()
    print(f"{f0:7d}   sin | rms b0 {np.sqrt((b[0]**2).mean()):8.4f}")

# --- real bat calls: how much energy actually lives in each band? ---
D = os.path.dirname(os.path.abspath(__file__))
sub = pd.read_csv(os.path.join(D, "bats", "subset2000.csv"))
print("\n=== real bat calls: native spectrum energy per 8 kHz band (pre-transform, dB rel. total) ===")
accum = np.zeros(5)
n = 0
for f in sub["File Name"].tolist()[:25]:
    p = os.path.join(D, "bats", "audio", f)
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    x = x[: int(6.0 * SR)]
    S = np.abs(np.fft.rfft(x)) ** 2
    fr = np.fft.rfftfreq(len(x), 1 / SR)
    tot = S.sum()
    for k in range(4):
        accum[k] += S[(fr >= 8000 * k) & (fr < 8000 * (k + 1))].sum() / tot
    accum[4] += S[fr >= 32000].sum() / tot
    n += 1
accum /= n
for k in range(4):
    print(f"  band {k} [{8*k:2d}-{8*(k+1):2d} kHz]: {100*accum[k]:6.2f}% of energy  ({10*np.log10(accum[k]+1e-20):+6.1f} dB)")
print(f"  >32 kHz (discarded): {100*accum[4]:6.2f}%  ({10*np.log10(accum[4]+1e-20):+6.1f} dB)")

print("\n=== post-transform band RMS on real calls (what the encoder actually sees) ===")
tot = np.zeros(4); n = 0
for f in sub["File Name"].tolist()[:15]:
    p = os.path.join(D, "bats", "audio", f)
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    x = x[: int(6.0 * SR)]
    b = t(torch.from_numpy(x).unsqueeze(0))[0].numpy()
    tot += np.sqrt((b ** 2).mean(1)); n += 1
tot /= n
for k in range(4):
    print(f"  band {k}: rms {tot[k]:.6f}  ({20*np.log10(tot[k]/tot[0]):+6.1f} dB rel. band 0)")
