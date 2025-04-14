# nyquist_vs_sparse_sampling
# Objective:
# To simulate the sampling of continuous-time signals using Python, and visually demonstrate
# the principles of Nyquist sampling and the consequences of undersampling (aliasing) when the 
# Nyquist criterion is violated.
# Background:
# The Nyquist-Shannon Sampling Theorem states that a continuous-time signal can be perfectly
#  reconstructed from its samples if it is band-limited and sampled at a rate at least twice 
# its maximum frequency (the Nyquist Rate). When this condition is not met, aliasing occurs — 
# different frequency components become indistinguishable, resulting in signal distortion.
# Tasks:
# Generate a set of continuous-time sinusoidal signals of varying frequencies.
# Sample these signals at different sampling rates:
# Above the Nyquist rate (safe)
# Exactly at the Nyquist rate (edge case)
# Below the Nyquist rate (undersampling)
# Reconstruct the signals using interpolation and compare with the original.
# Plot time-domain and frequency-domain views to visually explain aliasing.
# Expected Outcomes:
# Clear visualizations of the original and sampled signals.
# Frequency-domain plots showing how aliasing folds frequencies into incorrect locations.
# Intuition on how sampling rate affects signal integrity.
# 📊 Bonus:
# Add sliders or interactive plots (e.g., using ipywidgets or matplotlib) to dynamically change 
# frequency and sampling rate for live aliasing demonstration.

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Parameters
T0 = 1.0                      # Signal period
omega0 = 2 * np.pi / T0       # Fundamental frequency
K = 10                        # Number of harmonics (on each side)
t = np.linspace(0, T0, 1000)  # Time domain over one period

# Generate Fourier coefficients a_k (example: decaying or random)
a_k = {k: np.exp(-0.5 * abs(k)) * np.exp(1j * np.pi * k / 4) for k in range(-K, K+1)}

# Compute x(t) as sum of a_k * exp(i * k * omega0 * t)
x_t = np.zeros_like(t, dtype=complex)
for k in range(-K, K+1):
    x_t += a_k[k] * np.exp(1j * k * omega0 * t)

A = 1          # Amplitude
f = 5          # Frequency (Hz)
phi = 0        # Phase
duration = 1   # seconds
fs_cont = 1000 # "continuous" sample rate
fs_sample = 100  # choose < 2f to show aliasing

# Time vectors
t_cont = np.linspace(0, duration, int(fs_cont * duration), endpoint=False)
t_sampled = np.linspace(0, duration, int(fs_sample * duration), endpoint=False)

# Signals
x_cont = A * np.sin(2 * np.pi * f * t_cont + phi)
x_sampled = A * np.sin(2 * np.pi * f * t_sampled + phi)

# Interpolation to show reconstruction
recon = interp1d(t_sampled, x_sampled, kind='cubic', fill_value="extrapolate")
x_recon = recon(t_cont)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(t_cont, x_cont, label='Original (continuous)', linewidth=2, alpha=0.7)
plt.stem(t_sampled, x_sampled, linefmt='r-', markerfmt='ro', basefmt=" ", label='Sampled')
plt.plot(t_cont, x_recon, '--', label='Reconstructed', color='green', alpha=0.6)
plt.title(f'Sinusoid Sampling - f={f}Hz, fs={fs_sample}Hz')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



# Plot real part of x(t)
plt.figure(figsize=(10, 5))
plt.plot(t, np.real(x_t), label='Re{x(t)}')
plt.title('Signal x(t) as a Fourier Series of Complex Exponentials')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()