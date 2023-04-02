import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Set the seed for reproducibility
np.random.seed(0)

# Create a time vector
time = np.linspace(0, 10, 1000)

# Generate the original sinusoidal signal
original_signal = np.sin(2 * np.pi * time)

# Generate the noisy signal
noisy_signal = original_signal + 0.5 * np.random.randn(1000)

# Find the peaks in the noisy signal
peaks, _ = find_peaks(noisy_signal, distance=75)
# Extract the peak values in the original and noisy signals
original_peak_values = original_signal[peaks]
noisy_peak_values = noisy_signal[peaks]

# Compute the errors (differences) between the peak values
errors = original_peak_values - noisy_peak_values

# Compute the root-mean-square error (RMSE)
rmse = np.sqrt(np.mean(errors**2))

# Plot the original and noisy signals and the detected peaks
plt.plot(time, original_signal, label='Original signal')
plt.plot(time, noisy_signal, label='Noisy signal')
plt.plot(time[peaks], noisy_signal[peaks], 'ro', ms=5, label='Detected peaks')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.show()

# Print the RMSE
print(f'Root-mean-square error (RMSE) between original and detected peaks: {rmse:.4f}')
