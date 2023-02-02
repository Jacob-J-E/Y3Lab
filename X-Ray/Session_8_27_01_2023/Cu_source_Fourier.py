import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
hep.style.use("ATLAS")

"""
Fourier Cleaning Algorithm
"""

# Load in data
data = pd.read_csv(r"X-Ray\Data\28-01-2023\Cu No Filter Energy-Angle run.csv",skiprows=0)
angle = data['Angle']
counts = data['Counts']
sin_angle = np.sin(angle * np.pi / 180)

def fourier_clean(x_data,y_data,threshold):
    """
    FT algorithm to clean data.
    Takes FFT of y-data, sets high freq data to 0, takes inverse FFT
    Input y-data, x-data, threshold.
    """
    
    size = len(x_data)
    fft = np.fft.fft(y_data,size)
    abs_fft = fft * np.conj(fft)/size
    signal_threshold = threshold
    print(max(abs_fft), "max")
    abs_fft_zero_arr = abs_fft > signal_threshold #array of 0 and 1 to indicate where data is greater than threshold
    fhat_clean = abs_fft_zero_arr * fft #used to retrieve the signal
    signal_filtered = np.fft.ifft(fhat_clean) #inverse fourier transform

    return signal_filtered



"""
Test FT cleaning algorithm on Cu energy-angle NaCl Power data
"""
filtered_count = fourier_clean(angle,counts,100)
fig, ax = plt.subplots(1,3)
ax[1].sharey(ax[0])
ax[0].plot(sin_angle,counts)
ax[1].plot(sin_angle,filtered_count)
ax[2].plot(sin_angle,np.abs(filtered_count-counts))
ax[0].set_ylabel(r"Counts $(s^{-1})$")
for ax in ax:
    ax.set_xlabel(r"$\sin(\theta)$ (No Units)")
plt.tight_layout()
plt.show()



"""
FT cleaning algorithm on NaCl Crystal data

Doesn't really work tbh. I need a more sophisticated algorithm.
"""
data = pd.read_csv(r"X-Ray\Data\16-01-2023\NaCl Full Data.csv",skiprows=0)
angle = data['angle']
counts = data['R_0 / 1/s']
sin_angle = np.sin(angle * np.pi / 180)
filtered_count = fourier_clean(angle,counts,5000)
fig, ax = plt.subplots(1,3)
ax[1].sharey(ax[0])
ax[0].plot(sin_angle,counts)
ax[1].plot(sin_angle,filtered_count)
ax[2].plot(sin_angle,np.abs(filtered_count-counts))
ax[0].set_ylabel(r"Counts $(s^{-1})$")
for ax in ax:
    ax.set_xlabel(r"$\sin(\theta)$ (No Units)")
plt.tight_layout()
plt.show()