import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import seaborn as sns
import scipy.optimize as spo
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
hep.style.use("ATLAS")


data = pd.read_csv(r"X-Ray\Data\16-01-2023\NaCl Full Data.csv",skiprows=0)
print(data)

angle = data['angle']
wav = data['wav / pm']
energy = data['E / keV']
count_0 = data['R_0 / 1/s']

wav = wav * 1e-12 

angle_max = 5.5
wav = wav[angle>angle_max]
count_0 = count_0[angle>angle_max]

fft = np.fft.fft(count_0)
fft_copy = fft
# fft = np.fft.fftshift(fft)
dx = np.diff(wav)[0]
freqs = np.fft.fftfreq(len(wav), d=dx)

yhat = savgol_filter(fft, 99, 15) 

inv_fft = np.fft.ifft(fft)
inv_filt = np.fft.ifft(yhat)

fig,ax = plt.subplots(1,4)
ax[0].plot(wav,count_0,label="Real Space")
ax[1].plot(freqs,np.real(fft),label="Fourier Space")
ax[1].plot(freqs,yhat,label="Low Pass Filter")
ax[2].plot(wav[inv_fft>0],inv_fft[inv_fft>0],label="Inverse of FT")
ax[3].plot(wav[inv_filt>0],inv_filt[inv_filt>0],label="Inverse of Filter FT")

ax[0].legend(loc="upper right")
ax[1].legend(loc="upper right")
ax[2].legend(loc="upper right")

ax[0].set_xlabel("Wavelength (m)")
ax[1].set_xlabel("Wavenumber (m^-1)")
ax[2].set_xlabel("Wavenumber (m^-1)")


ax[3].set_xlabel("Wavelength (nm)")
plt.show()

# ax[3].set_xlim(min(inv_x),max(inv_x))

# for x in x:
#     ax[1].axvline(x, color = 'black')
# for inv_x in inv_x:   
#     ax[3].axvline(inv_x, color = 'black')

# local_maxima = argrelextrema(yhat, np.greater)
# amplitudes = []
# x= []
# for i in local_maxima[0]:
#     amplitudes.append(yhat[i])
#     x.append(freqs[i])
# x = np.real(np.array(x))
# inv_x = np.array(2*np.pi / x)




