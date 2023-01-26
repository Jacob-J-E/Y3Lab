import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import scipy.signal as ssp
from scipy.signal import argrelextrema
hep.style.use("ATLAS")

powder_data = pd.read_csv(r"X-Ray\Data\26-01-2023\NaCl_powder.csv",skiprows=0)

angle = powder_data['angle']
wav = powder_data['wav']
energy = powder_data['E / keV']
no_filter_NaCl = np.array(powder_data['R_0 / 1/s'])
with_filter_NaCl = powder_data['R_1 / 1/s']
filter_transmission = powder_data['T_1 / %']

no_filter_NaCl_savgol = ssp.savgol_filter(no_filter_NaCl,11,3)
with_filter_NaCl_savgol = ssp.savgol_filter(with_filter_NaCl,11,3)



A = 564.02e-12
# ENERGY1 = 17.443e3*1.6e-19
# ENERGY2 = 19.651e3*1.6e-19
ENERGY1 = 17330*1.6e-19
ENERGY2 = 19549*1.6e-19
wav_1 = (6.63e-34 * 3e8) / ENERGY1
wav_2 = (6.63e-34 * 3e8) / ENERGY2


def calculate_angle(h,k,l,energy):
    wavelength = (6.63e-34 * 3e8)/energy
    sin_angle = np.sqrt((wavelength**2/(4*A**2))*(h**2+k**2+l**2))
    angle = np.arcsin(sin_angle)*180/np.pi
    return angle - 2

primitive = set()
bcc = set()
fcc_even = set()
fcc_odd = set()
hcp = set()

def bcc_check(h,k,l):
    if (h+k+l)%2 == 0:
        return True
    return False

def fcc_check_even(h,k,l):
    if (h%2 == 0 and k%2 == 0 and l%2 == 0):
        return True
    return False

def fcc_check_odd(h,k,l):
    if (h%2 == 1 and k%2 == 1 and l%2 == 1):
        return True
    return False

def hcp_check(h,k,l):
    if l%2 == 1 and h+2*k == 3*(h+k+l):
        return False
    return True

range_max = range(0,6)
range_one = range(0,6)
for h in range_max:
    for k in range_one:
        for l in range_one:
            angle1 = calculate_angle(h,k,l,ENERGY1)
            angle2 = calculate_angle(h,k,l,ENERGY2)
            primitive.add(angle1)
            primitive.add(angle2)
            if bcc_check(h,k,l):
                bcc.add(angle1)
                bcc.add(angle2)
            if fcc_check_even(h,k,l):
                fcc_even.add(angle1)
                fcc_even.add(angle2)
            if fcc_check_odd(h,k,l):
                fcc_odd.add(angle1)
                fcc_odd.add(angle2)    
            if hcp_check(h,k,l):
                hcp.add(angle1)
                hcp.add(angle2)

angle_upper_threshold = 30

primitive = [i for i in primitive if i <= angle_upper_threshold]
bcc = [i for i in bcc if i <= angle_upper_threshold]
fcc_even = [i for i in fcc_even if i <= angle_upper_threshold]
fcc_odd = [i for i in fcc_odd if i <= angle_upper_threshold]
hcp = [i for i in hcp if i <= angle_upper_threshold]

# print(len(primitive))
# print(len(bcc))

# for idx,x in enumerate(primitive):
#     if not(idx == len(primitive)-1):
#         plt.axvline(x, color = 'r')
#     else:
#         plt.axvline(x, color = 'r', label = 'primitive lattice')

# for idx,x in enumerate(bcc):
#     if not(idx == len(bcc)-1):
#         plt.axvline(x, color = 'b')
#     else:
#         plt.axvline(x, color = 'b', label = 'bcc lattice')




fig,ax = plt.subplots(2,1) 

for idx,x in enumerate(fcc_even):
    if not(idx == len(fcc_even)-1):
        ax[1].axvline(x, color = 'black')
    #else:
        ax[1].axvline(x, color = 'black')



ax[0].plot(angle,no_filter_NaCl,label="Unfiltered NaCl Power")
ax[0].plot(angle,with_filter_NaCl,label="Zr Filter NaCl Power")
ax[1].plot(angle,no_filter_NaCl_savgol,label="Savgol Unfiltered NaCl Power")
ax[1].plot(angle,with_filter_NaCl_savgol,label="Savgol Zr Filter NaCl Power")

for ax in ax:
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel(r"Count Rate $(s^{-1})$")
    ax.legend(loc="upper right")
    ax.grid()

plt.show()
no_filter_NaCl_savgol_splice = no_filter_NaCl_savgol[angle<20]
with_filter_NaCl_savgol_splice = with_filter_NaCl_savgol[angle<20]


# Filter is doing some strange things here. Consider making the filter more coarse, or just using the OG datasets.
local_maxima = argrelextrema(no_filter_NaCl_savgol_splice, np.greater)
amplitudes = []
peak_angles_no_filter= []
for i in local_maxima[0]:
    amplitudes.append(no_filter_NaCl_savgol_splice[i])
    peak_angles_no_filter.append(angle[i])

local_maxima = argrelextrema(with_filter_NaCl_savgol_splice, np.greater)
amplitudes = []
peak_angles_with_filter= []
for i in local_maxima[0]:
    amplitudes.append(with_filter_NaCl_savgol_splice[i])
    peak_angles_with_filter.append(angle[i])

print(peak_angles_no_filter)
print(peak_angles_with_filter)


# Need to classify the order of each peak to calculate a_0 ?
