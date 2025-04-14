import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- Signal Construction (Fourier Series of Complex Exponentials) ---

# Parameters
T0 = .01                          # Signal period (seconds)
omega0 = 2 * np.pi / T0          # Fundamental frequency
K = 100                           # Number of harmonics (positive + negative)
duration = 1.0                   # Total signal duration (seconds)
fs_cont = 1000                   # High-rate continuous "reference" sampling
fs_sample = 1000                  # Sampling rate (change to simulate aliasing)

# Time vectors
t_cont = np.linspace(0, duration, int(fs_cont * duration), endpoint=False)
t_sample = np.linspace(0, duration, int(fs_sample * duration), endpoint=False)

# Generate complex Fourier coefficients a_k
a_k = {k: np.exp(-0.5 * abs(k)) * np.exp(1j * np.pi * k / 4) for k in range(-K, K + 1)}
print(a_k[3])
# Function to build signal from Fourier series
def build_signal(t, a_k, omega0, K):
    x = np.zeros_like(t, dtype=complex)
    for k in range(-K, K + 1):                # ∑ k⊆Z
        x += a_k[k] * np.exp(1j * k * omega0 * t)  # a_k*e^(ikw_0t)
    return x

# Create the continuous and sampled signals
x_cont = build_signal(t_cont, a_k, omega0, K)
x_sample = build_signal(t_sample, a_k, omega0, K)

# Interpolate (reconstruct) the real part of the signal from samples
recon_interp = interp1d(t_sample, np.real(x_sample), kind='cubic', fill_value="extrapolate")
x_recon = recon_interp(t_cont)

# --- Plotting ---
plt.figure(figsize=(12, 6))
plt.plot(t_cont, np.real(x_cont), label='Original x(t) - Real Part', linewidth=2, alpha=0.8)
plt.stem(t_sample, np.real(x_sample), linefmt='r-', markerfmt='ro', basefmt=" ", label='Sampled')
plt.plot(t_cont, x_recon, '--', label='Reconstructed (Interpolated)', color='green', alpha=0.6)
plt.title(f'Sampling of Fourier Series Signal | Sampling Rate = {fs_sample} Hz')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()