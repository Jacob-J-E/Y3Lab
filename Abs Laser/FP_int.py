import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
import scipy.interpolate as spi
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
from sklearn.preprocessing import *
hep.style.use("CMS")

data = pd.read_csv(r"Abs Laser\Data\10-03-2023\NEW1B.CSV")
data_DB_free = pd.read_csv(r"Abs Laser\Data\10-03-2023\NEW1.CSV")

x_axis = data_DB_free['in s']
c1 = data_DB_free['C1 in V']
c2 = data_DB_free['C2 in V']
c3 = data_DB_free['C3 in V']
c4 = data_DB_free['C4 in V']

c1_B = data['C1 in V']
x2 = data['in s']

FP = np.array(c4)
FP_sav = savgol_filter(FP,window_length=151,polyorder=3)
max_ind = argrelextrema(FP_sav,np.greater)
peak_y = FP_sav[max_ind[0]]
peak_x = np.array(x_axis[max_ind[0]][peak_y > -0.055])
peak_y = peak_y[peak_y > -0.055]


spacing = np.diff(peak_x)
# plt.plot(x_axis,FP)
# plt.scatter(peak_x,peak_y,color='red',marker='o')
# plt.show()
# spacing_true = 1e-3
spacing_true = 1/11
x_corr = []
j= 0
print(f' len(peak_x) {len(peak_x)}')
for i in range(0,len(x_axis)-1):
    #print(j)
    if j+1 == len(peak_x)-1:
        break
    if x_axis[i+1] == peak_x[j+1]:
        j += 1
    x_corr.append(x_axis[i]+(peak_x[j+1]-peak_x[j]/spacing_true)*(x_axis[i+1]-x_axis[i]))
    

x_corr = np.array(x_corr)
print("New std ",np.std(np.diff(x_corr)))
print("Spacing", np.diff(x_corr))

plt.plot(x_axis,FP,label="OG")
plt.plot(x_corr,0.01+FP[:len(x_corr)],label="Shifted")
plt.plot(x_axis,c1)
plt.plot(x_corr,c1[:len(x_corr)])

plt.legend(loc='upper right')
plt.show()


fft_abs = np.fft.rfft(c1)
fft_abs = np.fft.fftshift(fft_abs)
dx_abs = np.diff(x_corr)[0]
freqs_abs = np.fft.rfftfreq(len(x_corr), d=dx_abs)

fft_fp = np.fft.rfft(FP)
fft_fp = np.fft.fftshift(fft_fp)
dx_fp = np.diff(x_corr)[0]
freqs_fp = np.fft.rfftfreq(len(x_corr), d=dx_fp)


plt.plot(freqs_abs,fft_abs[:len(freqs_abs)],label="Abs")
plt.plot(freqs_fp,fft_fp[:len(freqs_fp)],label="FP")
plt.legend(loc='upper right')
plt.show()
