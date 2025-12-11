"""
Sample code ChatGPT for auto-correlation, spectral entropy, dominant frequency
"""

import numpy as np
import pandas as pd
from scipy.signal import welch, correlate
from scipy.stats import entropy

# --------------------------------------------------
# Example setup: generate a synthetic periodic signal
# --------------------------------------------------
fs = 50  # sampling frequency in Hz
t = np.arange(0, 10, 1/fs)  # 10 seconds
freq = 2  # 2 Hz motion (e.g., steps per second)

# Simulated triaxial acceleration (simple harmonic + noise)
ax = 0.8 * np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
ay = 0.6 * np.sin(2 * np.pi * freq * t + np.pi/3) + 0.1 * np.random.randn(len(t))
az = 0.4 * np.sin(2 * np.pi * freq * t + np.pi/6) + 0.1 * np.random.randn(len(t))

# Combine into magnitude (orientation-invariant)
a_mag = np.sqrt(ax**2 + ay**2 + az**2)

# --------------------------------------------------
# 1️⃣ Autocorrelation-based periodicity
# --------------------------------------------------
acf = correlate(a_mag - np.mean(a_mag), a_mag - np.mean(a_mag), mode='full')
acf = acf[acf.size // 2:]  # keep non-negative lags
acf /= acf[0]  # normalize

# Find first peak (dominant period)
lag_peaks = np.diff(np.sign(np.diff(acf))) < 0
peak_indices = np.where(lag_peaks)[0]
if len(peak_indices) > 0:
    dominant_lag = peak_indices[0]
    dominant_period = dominant_lag / fs
    periodicity_strength = acf[dominant_lag]
else:
    dominant_period = np.nan
    periodicity_strength = np.nan

print(f"Autocorrelation peak lag: {dominant_lag} samples ({dominant_period:.3f} s)")
print(f"Autocorrelation strength: {periodicity_strength:.3f}")

# --------------------------------------------------
# 2️⃣ Spectral entropy (lower = more periodic)
# --------------------------------------------------
f, psd = welch(a_mag, fs=fs, nperseg=1024)
psd_norm = psd / np.sum(psd)
spec_entropy = entropy(psd_norm, base=2)

print(f"Spectral entropy: {spec_entropy:.3f} bits")

# --------------------------------------------------
# 3️⃣ Dominant frequency
# --------------------------------------------------
dominant_freq = f[np.argmax(psd)]
print(f"Dominant frequency: {dominant_freq:.3f} Hz")

# --------------------------------------------------
# Optional: visualize
# --------------------------------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(t, a_mag)
plt.title("Acceleration magnitude")
plt.xlabel("Time (s)")
plt.ylabel("|a| (g)")

plt.subplot(1, 2, 2)
plt.semilogy(f, psd)
plt.title("Power Spectral Density")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.tight_layout()
plt.show()
