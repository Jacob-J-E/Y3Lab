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
# hep.style.use("CMS")

data = pd.read_csv(r"Abs_Laser\Data\10-03-2023\NEW1B.CSV")
data_DB_free = pd.read_csv(r"Abs_Laser\Data\10-03-2023\NEW1.CSV")

# data = pd.read_csv(r"Abs_Laser\Data\14-03-2023\DUB03B.CSV")
# data_DB_free = pd.read_csv(r"Abs_Laser\Data\14-03-2023\DUB03.CSV")

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
spacing_true = 1e-3
spacing_true = (max(peak_x[:-3])-min(peak_x[:-3]))/len(peak_x)
print("New spacing?? ", spacing_true)
x_corr = []
j= 0
print("Old std ",np.std(np.diff(peak_x)))
print("Old Spacing", np.diff(peak_x))
print(f' len(peak_x) {len(peak_x)}')

for i in range(0,len(x_axis)-1):
    #print(j)
    if j+1 == len(peak_x)-1:
        break
    if x_axis[i+1] == peak_x[j+1]:
        j += 1
    x_corr.append(x_axis[i]+(peak_x[j+1]-peak_x[j]/spacing_true)*(x_axis[i+1]-x_axis[i]))
    

x_corr = np.array(x_corr)
x_axis = np.array(x_axis)
print("New std ",np.std(np.diff(x_corr)))
print("Spacing", np.diff(x_corr))


# plt.plot(x_axis,FP,label="OG")
# plt.scatter(peak_x,peak_y,marker='o',color='r')
# plt.plot(x_corr,0.01+FP[:len(x_corr)],label="Shifted")
# plt.plot(x_axis,c1)
# plt.plot(x_corr,c1[:len(x_corr)])
# plt.plot(x_corr,c1_B[:len(x_corr)])

# plt.legend(loc='upper right')
# plt.show()


# print(len(x_corr[(peak_x[0] < x_corr) & (peak_x[1] > x_corr)]))
# print(len(x_corr[(peak_x[1] < x_corr) & (peak_x[2] > x_corr)]))
# print(len(x_corr[(peak_x[2] < x_corr) & (peak_x[3] > x_corr)]))
# print(len(x_corr[(peak_x[3] < x_corr) & (peak_x[4] > x_corr)]))

# def frequency_scale(x,delta_f,detla_t)
# def f(x):
#     return 1.6163e15 * x + 4.05e14

def lin_int(x,x_0,x_1,y_0,y_1):
    return y_0 * ((x_1-x)/(x_1-x_0)) + y_1 * ((x-x_0)/(x_1-x_0))


true_freq_0 = 3e8/780e-9 - 1776e6
true_freq_1 = 3e8/780e-9 + 1269e6
# true_freq_0 = 0 + 2.8e9
# true_freq_1 = 0 + 1269e6-1776e6

x_0 = -0.0030069
x_1 = 0.00693048

freq = lin_int(x_corr,x_0,x_1,true_freq_0,true_freq_1)

# vals = []
# for i in range(0,len(peak_x)-4):
#     diff = len(x_corr[(peak_x[i] < x_corr) & (peak_x[i+1] > x_corr)])
#     print(diff)
#     vals.append(diff)


# print("std ",np.std(np.array(vals)))
# print("range ",max(vals)-min(vals))
fig,ax = plt.subplots(1,2)

ax[0].plot(freq,c1[:len(x_corr)])
ax[0].plot(freq,c1_B[:len(x_corr)])
ax[0].plot(freq,FP[:len(x_corr)])
c1-c1_B/max(c1_B)*max(c1)
ax[1].plot(freq, (c1[:len(x_corr)] - c1_B[:len(x_corr)])/(max(c1_B[:len(x_corr)]))*max(c1[:len(x_corr)]))
plt.show()


def line(x,m,c):
    return m * x + c

import scipy.optimize as spo
x_axis = np.array(x_axis)
x_corr = np.array(x_corr)
# x_plot = x_axis[:len(freq)]
x_plot = x_corr
x_axis_len = x_axis[:len(freq)]

init_guess = [(freq[10]-freq[7])/(x_plot[10]-x_plot[7]),3e8/780e-9]
params,cov = spo.curve_fit(line,x_corr,freq,p0=init_guess)


uncor_guess = [(freq[10]-freq[7])/(x_axis_len[10]-x_axis_len[7]),3e8/780e-9]
uncor_params, uncor_cov = spo.curve_fit(line,x_axis_len,freq,p0=uncor_guess)
print("Params: ",params)
print("Params Uncor: ",uncor_params)


def chi_sq(E,O):
    return sum((E-O)**2/E)

print("Uncorrected Chi^2: ",chi_sq(line(x_axis_len,*uncor_params),freq))
print("FP Corrected Chi^2: ",chi_sq(line(x_corr,*params),freq))

plt.scatter(x_plot,freq,alpha=0.5)
plt.plot(x_plot,line(x_plot,*params),color='red')
plt.xlabel("Recorded Time (s)")
plt.ylabel("Frequency (Hz)")
plt.show()






# def line(x,m,c):
#     return m * x + c

# # x_plot = x_axis[:len(freq)]
# x_plot = x_corr[:len(freq)]


# init_guess = [(freq[10]-freq[7])/(x_plot[10]-x_plot[7]),3e8/780e-9]
# params,cov = spo.curve_fit(line,x_plot,freq,p0=init_guess)
# print("Params: ",params)

# def chi_sq(E,O):
#     return sum((E-O)**2/E)
# # plt.scatter(freq,x_axis[:len(freq)])
# print("Uncorrected Chi: ",chi_sq(line(x_axis[:len(freq)],*params),freq))
# print("FP Corrected Chi: ",chi_sq(line(x_plot,*params),freq))

# # plt.scatter(x_plot,freq,alpha=0.5)
# plt.scatter(x_axis[:len(freq)],freq)
# # plt.scatter(x_axis[:len(freq)],freq,alpha=0.5)

# # plt.plot(x_plot,line(x_plot,*params),color='red')
# plt.xlabel("Recorded Time (s)")
# plt.ylabel("Frequency (Hz)")
# plt.show()






















# fft_abs = np.fft.fft(c1)
# fft_abs = np.fft.fftshift(fft_abs)
# dx_abs = np.diff(x_corr)[0]
# freqs_abs = np.fft.fftfreq(len(x_corr), d=dx_abs)

# fft_fp = np.fft.fft(FP)
# fft_fp = np.fft.fftshift(fft_fp)
# dx_fp = np.diff(x_corr)[0]
# freqs_fp = np.fft.fftfreq(len(x_corr), d=dx_fp)


# plt.plot(freqs_abs,fft_abs[:len(freqs_abs)],label="Abs")
# plt.plot(freqs_fp,fft_fp[:len(freqs_fp)],label="FP")
# plt.legend(loc='upper right')
# plt.show()
