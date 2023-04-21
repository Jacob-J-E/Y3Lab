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
import scipy.optimize as spo
# def lin_int(x,x_0,x_1,y_0,y_1):
#     return y_0 * ((x_1-x)/(x_1-x_0)) + y_1 * ((x-x_0)/(x_1-x_0))

# def freq_scale(data,data_B):
#     x_axis = data['in s']
#     c1 = data['C1 in V']
#     # c2 = data['C2 in V'] 
#     # c3 = data['C3 in V']
#     c4 = data['C4 in V']

#     c1_B = data_B['C1 in V']
#     # x2 = data_B['in s']

#     FP = np.array(c4)
#     FP_sav = savgol_filter(FP,window_length=151,polyorder=3)
#     max_ind = argrelextrema(FP_sav,np.greater)
#     peak_y = FP_sav[max_ind[0]]
#     peak_x = np.array(x_axis[max_ind[0]][peak_y > -0.055])
#     peak_y = peak_y[peak_y > -0.055]

#     # spacing = np.diff(peak_x)
#     spacing_true = (max(peak_x[:-3])-min(peak_x[:-3]))/len(peak_x)

#     x_corr = []
#     j= 0    

#     for i in range(0,len(x_axis)-1):
#         if j+1 == len(peak_x)-1:
#             break
#         if x_axis[i+1] == peak_x[j+1]:
#             j += 1
#         x_corr.append(x_axis[i]+(peak_x[j+1]-peak_x[j]/spacing_true)*(x_axis[i+1]-x_axis[i]))
    
#     x_corr = np.array(x_corr)
#     x_axis = np.array(x_axis)

#     true_freq_0 = 3e8/780e-9 - 1776e6
#     true_freq_1 = 3e8/780e-9 + 1269e6

#     x_0 = -0.0030069
#     x_1 = 0.00693048

#     freq = lin_int(x_corr,x_0,x_1,true_freq_0,true_freq_1)

#     return (freq, (-c1[:len(x_corr)] + c1_B[:len(x_corr)])/(max(c1_B[:len(x_corr)]))*max(c1[:len(x_corr)]))


freqs = []
hfs = []

low_cut = 3.84614336e14
high_cut = 3.84616487e14

data_0 = pd.read_csv(r"Abs_Laser\Data\17-03-2023\ZEEMAN"+str(0)+".CSV")
data_0_B = pd.read_csv(r"Abs_Laser\Data\17-03-2023\ZEEMAN"+str(0)+"B.CSV")
mag_field = pd.read_csv(r"Abs_Laser\Data\21-03-2023\B_Field.csv")
mag_field = np.array(mag_field['B'])
B_field = []
for i in range(len(mag_field)):
    if i%2 == 0:
        B_field.append(mag_field[i])
B_field = np.array(B_field)
mag_field = B_field
mag_field = mag_field -6.93
print("AHHH MAGNETIC FIELD",B_field)
# find_peaks

fig,ax = plt.subplots(1,2)
# freq_0, hfs_0 = freq_scale(data_0,data_0_B)
# ax[0].plot(freq_0[(freq_0 > low_cut) & (freq_0 < high_cut)],-hfs_0[(freq_0 > low_cut) & (freq_0 < high_cut)],label=r"$v_{app}= $"+str(0)+r"$V$")


peak_x = []
peak_y = []
freq_min = -0.008
for i in range(1,60):
    # data = pd.read_csv(r"Abs_Laser\Data\17-03-2023\ZEEMAN"+str(i)+".CSV")
    # data_B = pd.read_csv(r"Abs_Laser\Data\17-03-2023\ZEEMAN"+str(i)+"B.CSV")
    # data_B = pd.read_csv(r"Abs_Laser\Data\17-03-2023\ZEEMAN"+str(3*i)+".CSV")
    # data = pd.read_csv(r"Abs_Laser\Data\21-03-2023\ZR"+str(3*i)+".CSV")
    # data_B = pd.read_csv(r"Abs_Laser\Data\21-03-2023\ZB"+str(3*i)+".CSV")

    data = pd.read_csv(r"Abs_Laser\Data\21-03-2023\Z"+str(i)+".CSV")
    data_B = pd.read_csv(r"Abs_Laser\Data\21-03-2023\ZB"+str(i)+".CSV")



    # freq, hf = freq_scale(data,data_B)

    # hf = data['C1 in V']
    c1 = data['C1 in V']
    c1_B = data_B['C1 in V']
    freq = data['in s']

    c1 = c1[(freq>-0.015) & (freq < 0.008)]
    c1_B = c1_B[(freq>-0.015) & (freq < 0.008)]
    freq = freq[(freq>-0.015) & (freq < 0.008)]

    hf=  (-c1 + c1_B)/(max(c1_B))*max(c1)

  

    hf = np.array(hf)
    freq = np.array(freq)
    

    # hf = hf[(freq < 0.048) & (freq > 0.042)]
    # freq = freq[(freq < 0.048) & (freq > 0.042)]

    # hf = hf[(freq < 0.0500) & (freq > 0.0400)]
    # freq = freq[(freq < 0.0500) & (freq > 0.0400)]

    # hf = hf[(freq < 0.01475) & (freq > 0.01275)]
    # freq = freq[(freq < 0.01475) & (freq > 0.01275)]



    # hf_cut = hf[(freq > low_cut) & (freq < high_cut)]
    # freq_cut = freq[(freq > low_cut) & (freq < high_cut)]
    hf_cut = hf
    freq_cut = freq

    hf_smooth = -hf_cut+i


    freqs.append(freq)
    hfs.append(hf)
    c1_B = np.array(c1_B)

    CB_plot = -c1_B*(1+0.2*i)+i/4
    C1_plot = -c1*(1+0.2*i)+i/4
    HF_plot = -hf*(1+0.2*i)+i/4

    CB_peaks = CB_plot[freq>freq_min]
    freq_peaks = freq[freq>freq_min]
    max_ind_cb = find_peaks(CB_peaks,prominence=0.1)

    cb_x_peak = freq_peaks[max_ind_cb[0]]
    cb_y_peak = CB_peaks[max_ind_cb[0]]

    if mag_field[i] > 0:
        if i%4 == 0:
            ax[0].plot(freq,CB_plot,label=r"$v_{app}= $"+str(i)+r"$V$",color='black')
            ax[0].scatter(cb_x_peak,cb_y_peak,color='red')

        for point in cb_x_peak:
            ax[1].scatter(mag_field[i],point,color='blue',marker='x')
            peak_y.append(point)
            peak_x.append(mag_field[i])
            # ax[1].scatter(i*0.8,point,color='blue',marker='x')
            # peak_y.append(point)
            # peak_x.append(mag_field[2*i+1]-7)


ax[0].axvline(freq_min)
ax[0].set_xlabel("Frequency (Hz)")
ax[0].set_ylabel("Voltage")

ax[1].set_xlabel("Applied Magnetic Field (mT)")
ax[1].set_ylabel("Frequency (Hz)")

ax[0].set_title("Peak Structure")
ax[1].set_title("Zeeman Splitting")

plt.show()

plt.scatter(peak_x,peak_y)
plt.show()



    # ax[0].plot(freq_cut,hf_smooth,color='black')
    # max_ind = argrelextrema(hf_smooth,np.greater,order=3)
    # max_ind = find_peaks(hf_smooth,distance=200,prominence=10)


    # print(max_ind)
    # print("///")
    # print(max_ind[0])
    # peak_y = hf_smooth[max_ind[0]]
    # peak_x = freq_cut[max_ind[0]]

    # peak_y = list(peak_y)
    # peak_x = list(peak_x)

    # length = len(peak_y)
    # new_peak_y = []
    # new_peak_x = []

    # for j in range(0,length-1):
    #     if peak_y[j] > i/4:
    #         new_peak_y.append(peak_y[j])
    #         new_peak_x.append(peak_x[j])

    # hf = savgol_filter(-hf_cut,window_length=301,polyorder=3)

