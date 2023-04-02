import numpy as np

import numpy as np
from scipy.fft import fft, ifft
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, unit_impulse
from scipy.ndimage import convolve1d

def deconvolve_rc_filter(convolved_signal, frequencies, R, C, noise_power):
    # Compute the frequency response of the RC filter
    H = rc_filter_response(frequencies, R, C)

    # Compute the Fourier transform of the convolved signal
    convolved_signal_fft = fft(convolved_signal)

    # Apply the Wiener deconvolution
    wiener_filter = np.conj(H) / (np.abs(H) ** 2 + noise_power)
    recovered_signal_fft = convolved_signal_fft * wiener_filter

    # Compute the inverse Fourier transform to obtain the recovered signal
    recovered_signal = np.real(ifft(recovered_signal_fft))

    return recovered_signal


def rc_filter_response(frequencies, R, C):
    """
    Calculate the frequency response of an RC filter.
    
    Parameters:
    frequencies (array): An array of frequencies at which to compute the response.
    R (float): Resistance in ohms.
    C (float): Capacitance in farads.
    
    Returns:
    array: Frequency response of the RC filter at the input frequencies.
    """
    # Calculate the time constant
    tau = R * C
    
    # Calculate the angular frequencies
    omega = 2 * np.pi * frequencies
    
    # Compute the frequency response
    response = 1 / (1 + 1j * omega * tau)
    
    return response


import matplotlib.pyplot as plt

# Generate an example signal
t = np.linspace(0, 1, 1000)
original_signal = np.sin(2 * np.pi * 10 * t)

# Define RC parameters
R = 1e3  # 1 kOhm
C = 1e-9  # 1 nF

# Generate an impulse response of the RC filter
dt = t[1] - t[0]
frequencies = np.fft.fftfreq(len(t), dt)
impulse_response = np.real(ifft(rc_filter_response(frequencies, R, C)))

# Convolve the original signal with the impulse response
convolved_signal = convolve1d(original_signal, impulse_response, mode='reflect')

# Deconvolve the dataset using the RC filter model
noise_power = 1e-4
recovered_signal = deconvolve_rc_filter(convolved_signal, frequencies, R, C, noise_power)

# Plot the original, convolved, and recovered signals
plt.figure(figsize=(12, 6))
plt.plot(t, original_signal, label='Original Signal')
plt.plot(t, convolved_signal, label='Convolved Signal', alpha=0.5)
plt.plot(t, recovered_signal, label='Recovered Signal (Deconvolution)', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.show()