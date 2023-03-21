import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
import scipy.interpolate as spi
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
from scipy.signal import deconvolve
from scipy.signal import convolve
from sklearn.preprocessing import *

def Gauss(x,A,mu,sigma):
    return A * np.exp(-(x-mu)**2/(2*sigma**2))

def wiener_filter(img, kernal, k):
    kernal /= np.sum(kernal)
    dummy = np.copy(img)
    dummy = np.fft.rfft(img)
    kernal = np.fft.rfft(kernal)
    kernal = np.conj(kernal) / (np.abs(kernal)**2  + k)
    dummy = dummy * kernal
    dummy = np.abs(np.fft.irfft(dummy))
    return dummy

data = pd.read_csv(r"Abs_Laser\Data\10-03-2023\NEW1B.CSV")
data_DB_free = pd.read_csv(r"Abs_Laser\Data\10-03-2023\NEW1.CSV")

x_axis = data_DB_free['in s']
c1 = data_DB_free['C1 in V']
c2 = data_DB_free['C2 in V'] 
c3 = data_DB_free['C3 in V']
c4 = data_DB_free['C4 in V']

c1_B = data['C1 in V']
x2 = data['in s']
import copy
new_int = c1 - c1_B/(max(c1_B))*max(c1)

# sigma = 0.00002
sigma = 0.00002


blur = Gauss(x_axis,A=0.3,mu=0.0111,sigma=sigma)
blur_copy = blur.copy()

# noisy_data = np.convolve(new_int,blur_copy,mode='same')
# noisy_data = noisy_data/max(noisy_data)*max(new_int)
# filtered = wiener_filter(noisy_data,blur_copy,k=1)
# filtered = filtered/max(filtered)*max(new_int)


# plt.plot(x_axis,new_int,color='blue',label='Data')
# plt.plot(x_axis,blur,color='orange',label='Blur')
# plt.plot(x_axis,noisy_data,color='green',label="Noisy Data")
# plt.plot(x_axis,filtered,color='black',label="Fixed Data")
# plt.legend(loc='upper right')


new_data = wiener_filter(new_int,blur_copy,k=0.01)
plt.plot(x_axis,new_data/max(new_data))
plt.plot(x_axis+0.02,new_int/max(new_int)+0.0)
plt.xlim(0.025,0.029)
plt.ylim(0,1.2)
plt.show()

