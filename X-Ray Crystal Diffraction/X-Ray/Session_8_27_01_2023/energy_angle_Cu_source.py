import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
hep.style.use("CMS")
from scipy.signal import argrelextrema

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
    abs_fft_zero_arr = abs_fft > signal_threshold #array of 0 and 1 to indicate where data is greater than threshold
    fhat_clean = abs_fft_zero_arr * fft #used to retrieve the signal
    signal_filtered = np.fft.ifft(fhat_clean) #inverse fourier transform

    return signal_filtered


data = pd.read_csv(r"X-Ray\Data\28-01-2023\Cu No Filter Energy-Angle run.csv",skiprows=0)

angle = data['Angle']
counts = data['Counts']
cleaned_counts = fourier_clean(angle,counts,100)
# Find Maxima

local_maxima = argrelextrema(np.array(counts), np.greater)
amplitudes = []
x= []
for i in local_maxima[0]:
    if np.array(counts)[i] > 20:
        amplitudes.append(np.array(counts)[i])
        x.append(angle[i])
x = np.real(np.array(x))


local_maxima_clean = argrelextrema(np.array(cleaned_counts), np.greater)
amplitudes_clean = []
x_clean= []
for i in local_maxima_clean[0]:
    if np.array(cleaned_counts)[i] > 15:
        amplitudes_clean.append(np.array(cleaned_counts)[i])
        x_clean.append(angle[i])
x_clean = np.real(np.array(x_clean))

fig,ax = plt.subplots(1,2)
ax[0].plot(angle,counts,label="Experimental Data")
ax[0].set_xlabel("Angle (Degrees)")
ax[0].set_ylabel(r"Counts $(s^{-1})$")
ax[0].scatter(x,amplitudes,color='red')
ax[0].set_title("Experimental Data")

ax[1].plot(angle,cleaned_counts,label="Fourier Cleaned Data")
ax[1].set_xlabel("Angle (Degrees)")
ax[1].set_ylabel(r"Counts $(s^{-1})$")
ax[1].scatter(x_clean,amplitudes_clean,color='red')
ax[1].set_title("Fourier Cleaned Data")
plt.show()