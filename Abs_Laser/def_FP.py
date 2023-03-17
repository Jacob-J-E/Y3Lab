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

def lin_int(x,x_0,x_1,y_0,y_1):
    return y_0 * ((x_1-x)/(x_1-x_0)) + y_1 * ((x-x_0)/(x_1-x_0))

def freq_scale(data,data_B):
    x_axis = data['in s']
    c1 = data['C1 in V']
    # c2 = data['C2 in V'] 
    # c3 = data['C3 in V']
    c4 = data['C4 in V']

    c1_B = data_B['C1 in V']
    # x2 = data_B['in s']

    FP = np.array(c4)
    FP_sav = savgol_filter(FP,window_length=151,polyorder=3)
    max_ind = argrelextrema(FP_sav,np.greater)
    peak_y = FP_sav[max_ind[0]]
    peak_x = np.array(x_axis[max_ind[0]][peak_y > -0.055])
    peak_y = peak_y[peak_y > -0.055]

    # spacing = np.diff(peak_x)
    spacing_true = (max(peak_x[:-3])-min(peak_x[:-3]))/len(peak_x)

    x_corr = []
    j= 0    

    for i in range(0,len(x_axis)-1):
        if j+1 == len(peak_x)-1:
            break
        if x_axis[i+1] == peak_x[j+1]:
            j += 1
        x_corr.append(x_axis[i]+(peak_x[j+1]-peak_x[j]/spacing_true)*(x_axis[i+1]-x_axis[i]))
    
    x_corr = np.array(x_corr)
    x_axis = np.array(x_axis)

    true_freq_0 = 3e8/780e-9 - 1776e6
    true_freq_1 = 3e8/780e-9 + 1269e6

    x_0 = -0.0030069
    x_1 = 0.00693048

    freq = lin_int(x_corr,x_0,x_1,true_freq_0,true_freq_1)

    return (freq, (-c1[:len(x_corr)] + c1_B[:len(x_corr)])/(max(c1_B[:len(x_corr)]))*max(c1[:len(x_corr)]))


freqs = []
hfs = []

low_cut = 3.84614336e14
high_cut = 3.84616487e14

data_0 = pd.read_csv(r"Abs_Laser\Data\17-03-2023\ZEEMAN"+str(0)+".CSV")
data_0_B = pd.read_csv(r"Abs_Laser\Data\17-03-2023\ZEEMAN"+str(0)+"B.CSV")



fig,ax = plt.subplots(1,2)
freq_0, hfs_0 = freq_scale(data_0,data_0_B)
ax[0].plot(freq_0[(freq_0 > low_cut) & (freq_0 < high_cut)],-hfs_0[(freq_0 > low_cut) & (freq_0 < high_cut)],label=r"$v_{app}= $"+str(0)+r"$V$")

for i in range(1,16):
    data = pd.read_csv(r"Abs_Laser\Data\17-03-2023\ZEEMAN"+str(i)+".CSV")
    data_B = pd.read_csv(r"Abs_Laser\Data\17-03-2023\ZEEMAN"+str(i)+"B.CSV")
    freq, hf = freq_scale(data,data_B)

    hf = np.array(hf)

    hf_cut = hf[(freq > low_cut) & (freq < high_cut)]
    freq_cut = freq[(freq > low_cut) & (freq < high_cut)]

    hf_smooth = savgol_filter(-hf_cut+i/4,window_length=601,polyorder=3)

    ax[0].plot(freq_cut,hf_smooth,color='black')
    max_ind = argrelextrema(hf_smooth,np.greater,order=3)
    # print(max_ind)
    # print("///")
    # print(max_ind[0])
    peak_y = hf_smooth[max_ind[0]]
    peak_x = freq_cut[max_ind[0]]

    peak_y = list(peak_y)
    peak_x = list(peak_x)

    length = len(peak_y)
    new_peak_y = []
    new_peak_x = []
    for j in range(0,length-1):
        if peak_y[j] > i/4:
            new_peak_y.append(peak_y[j])
            new_peak_x.append(peak_x[j])

   

    freqs.append(freq)
    hfs.append(hf)

    # plt.plot(freq_cut,-hf_cut+i/4,label=r"$v_{app}= $"+str(i)+r"$V$")
    ax[0].scatter(new_peak_x,new_peak_y,marker='o',color='red')
    ax[1].scatter(np.zeros_like(new_peak_x)+i,new_peak_x)


ax[0].set_xlabel("Frequency (Hz)")
ax[0].set_ylabel("Voltage")

ax[1].set_xlabel("Applied Magnetic Field (mT)")
ax[1].set_ylabel("Frequency (Hz)")
plt.show()








