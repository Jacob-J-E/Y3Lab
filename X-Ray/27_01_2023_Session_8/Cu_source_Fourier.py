import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
hep.style.use("CMS")

"""
Fourier Cleaning Algorithm
"""

data = pd.read_csv(r"X-Ray\Data\28-01-2023\Cu No Filter Energy-Angle run.csv",skiprows=0)
angle = data['Angle']
counts = data['Counts']

sin_angle = np.sin(angle * np.pi / 180)

dt = 0.001
n = len(sin_angle)
fft = np.fft.fft(counts,n)
psd = fft * np.conj(fft)/n
freq = (1/(dt*n)) * np.arange(n) #frequency array
idxs_half = np.arange(1, np.floor(n/2), dtype=np.int32)
threshold = 20
psd_idxs = psd > threshold #array of 0 and 1
psd_clean = psd * psd_idxs #zero out all the unnecessary powers
fhat_clean = psd_idxs * fft #used to retrieve the signal

signal_filtered = np.fft.ifft(fhat_clean) #inverse fourier transform


fig, ax = plt.subplots(1,3)
ax[0].plot(sin_angle,counts)
ax[1].plot(sin_angle,signal_filtered)
ax[2].plot(sin_angle,np.abs(signal_filtered-counts))
plt.show()

