import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.signal import savgol_filter

hep.style.use("ATLAS")

data_NaCl = pd.read_csv(r"X-Ray\Data\20-01-2023\NaCl Baseline.csv",skiprows=0)
data_Ag = pd.read_csv(r"X-Ray\Data\20-01-2023\Ag Full Data.csv",skiprows=0)
data_Al = pd.read_csv(r"X-Ray\Data\20-01-2023\Al Full Data.csv",skiprows=0)
data_Mo = pd.read_csv(r"X-Ray\Data\20-01-2023\Mo Full Data.csv",skiprows=0)
data_Zr = pd.read_csv(r"X-Ray\Data\20-01-2023\Zr Full Data.csv",skiprows=0)

# These variables are the same for all elements.
angle = data_NaCl['angle']
wav = data_NaCl['n&l / pm']
energy = data_NaCl['E / keV']


# Load in data for all four elements
count_Zr = data_Zr['R_1 / 1/s']
T_Zr =  data_Zr['T_1 / %']

count_Ag = data_Ag['R_2 / 1/s']
T_Ag = data_Ag['T_2 / %']

count_Mo = data_Mo['R_3 / 1/s']
T_Mo = data_Mo['T_3 / %']

count_Al = data_Al['R_4 / 1/s']
T_Al = data_Al['T_4 / %']


"""
Generate initial plots of Count Rate
"""
# fig,ax = plt.subplots(2,2, sharex=True, sharey=True)

# ax[0][0].plot(wav,count_Zr,label="Zr Data")
# ax[0][1].plot(wav,count_Ag,label="Ag Data")
# ax[1][0].plot(wav,count_Mo,label="Mo Data")
# ax[1][1].plot(wav,count_Al,label="Al Data")

# ax[0][0].set_title("Zr Filter")
# ax[0][1].set_title("Ag Filter")
# ax[1][0].set_title("Mo Filter")
# ax[1][1].set_title("Al Filter")


# ax[1][0].set_xlabel(r"Wavelength /$pm$")
# ax[1][1].set_xlabel(r"Wavelength /$pm$")
# ax[0][0].set_ylabel("Count Rate /s")
# ax[1][0].set_ylabel("Count Rate /s")

# for i in range(0,2):
#     for j in range(0,2):
#         ax[i][j].legend(loc="upper right")
#         ax[i][j].grid()
#         # Figure out code to title all graphs

# fig.suptitle("Wavelength Spectra of NaCl With Various Metal Filters")
# plt.tight_layout()
# plt.show()

"""
Generate initial plots of transmission
"""
# Generate initial plots of Count Rate
fig,ax = plt.subplots(2,2, sharex=True, sharey=True)

Zr_filter = savgol_filter(T_Zr, 21, 3,mode='mirror') 
Ag_filter = savgol_filter(T_Ag, 21, 3,mode='mirror') 
Mo_filter = savgol_filter(T_Mo, 21, 3,mode='mirror') 
Al_filter = savgol_filter(T_Al, 21, 3,mode='mirror') 

ax[0][0].plot(wav,T_Zr,label="Zr Data")
ax[0][1].plot(wav,T_Ag,label="Ag Data")
ax[1][0].plot(wav,T_Mo,label="Mo Data")
ax[1][1].plot(wav,T_Al,label="Al Data")

ax[0][0].plot(wav,Zr_filter,label="Zr Filter",color='red')
ax[0][1].plot(wav,Zr_filter,label="Ag Filter",color='red')
ax[1][0].plot(wav,Zr_filter,label="Mo Filter",color='red')
ax[1][1].plot(wav,Zr_filter,label="Al Filter",color='red')

ax[0][0].set_title("Zr Filter")
ax[0][1].set_title("Ag Filter")
ax[1][0].set_title("Mo Filter")
ax[1][1].set_title("Al Filter")


ax[1][0].set_xlabel(r"Wavelength /$pm$")
ax[1][1].set_xlabel(r"Wavelength /$pm$")
ax[0][0].set_ylabel("Transmission /%")
ax[1][0].set_ylabel("Transmission /%")

for i in range(0,2):
    for j in range(0,2):
        ax[i][j].legend(loc="upper right")
        ax[i][j].grid()
        # Figure out code to title all graphs

fig.suptitle("Transmission of NaCl With Various Metal Filters")
plt.tight_layout()
plt.show()
