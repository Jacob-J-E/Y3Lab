import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.signal import savgol_filter
import scipy.optimize as spo
hep.style.use("CMS")

def line(x,m,c):
    return m*x+c

data_NaCl = pd.read_csv(r"X-Ray\Data\16-01-2023\NaCl Full Data.csv",skiprows=0)
data_Ag = pd.read_csv(r"X-Ray\Data\17-01-2023\Ag_Filter_NaCl.csv",skiprows=0)
data_Al = pd.read_csv(r"X-Ray\Data\17-01-2023\Al_Filter_NaCl.csv",skiprows=0)
data_Mo = pd.read_csv(r"X-Ray\Data\17-01-2023\Mo_Filter_NaCl.csv",skiprows=0)
data_Zr = pd.read_csv(r"X-Ray\Data\17-01-2023\ZR_Filter_NaCl.csv",skiprows=0)
# print(data_NaCl)

angle_Ag = data_Ag['angle']
angle_Al = data_Al['angle']
angle_Mo = data_Mo['angle']
angle_Zr = data_Zr['angle']

angle = np.array(data_NaCl['angle'])
wav = data_NaCl['wav / pm']


wav = wav[(angle>3) & (angle<15)]


wav_Ag = data_Ag['n&l / pm']
R_0 = data_NaCl['R_0 / 1/s']
R_0 = np.array(R_0[(angle>3) & (angle<15)])


R_Ag = data_Ag['R_0 / 1/s']
R_Al = data_Al['R_0 / 1/s']
R_Mo = data_Mo['R_0 / 1/s']
R_Zr = data_Zr['R_1 / 1/s']

angle = angle[(angle>3) & (angle<15)]

R_Ag = np.array(R_Ag[(angle_Ag>3) & (angle_Ag<15)])
R_Al = np.array(R_Al[(angle_Al>3) & (angle_Al<15)])
R_Mo = np.array(R_Mo[(angle_Mo>3) & (angle_Mo<15)])
R_Zr = np.array(R_Zr[(angle_Zr>3) & (angle_Zr<15)])

 
# print(len(angle))
# print(len(angle_Ag))

# print(len(wav))
# print(len(R_0))
# print(len(R_Ag))
# print(len(R_Ag/R_0))


# fig, ax = plt.subplots(2,2)
# ax[0][0].plot(wav,100*R_Ag/R_0)
# ax[0][1].plot(wav,100*R_Al/R_0)
# ax[1][0].plot(wav,100*R_Mo/R_0)
# ax[1][1].plot(wav,100*R_Zr/R_0)

# ax[0][0].set_xlabel("Wavelength (pm)")
# ax[0][1].set_xlabel("Wavelength (pm)")
# ax[1][0].set_xlabel("Wavelength (pm)")
# ax[1][1].set_xlabel("Wavelength (pm)")

# ax[0][0].set_ylabel("Transmission (%)")
# ax[0][1].set_ylabel("Transmission (%)")
# ax[1][0].set_ylabel("Transmission (%)")
# ax[1][1].set_ylabel("Transmission (%)")

# ax[0][0].grid()
# ax[0][1].grid()
# ax[1][0].grid()
# ax[1][1].grid()

# ax[0][0].set_title("Spectrum with Ag Filter")
# ax[0][1].set_title("Spectrum with Al Filter")
# ax[1][0].set_title("Spectrum with Mo Filter")
# ax[1][1].set_title("Spectrum with Zr Filter")
# plt.show()



Z = np.array([40,42,47])
wav_2 = np.array([71,61,46.5]) * 10e-12
wav_4 = np.array([142,125,95]) * 10e-12
grad_2 = (1/np.sqrt(wav_2[2])-1/np.sqrt(wav_2[0]))/(Z[2]-Z[0])
grad_4 = (1/np.sqrt(wav_4[2])-1/np.sqrt(wav_4[0]))/(Z[2]-Z[0])

print
print(f"Grad squared N=2: {grad_2**2}")
# print(f"Grad squared N=4: {line_4_params[0]**2}")

line_2_guess = [grad_2,1/np.sqrt(wav_2[1])-grad_2*Z[1]]
line_4_guess = [grad_4,1/np.sqrt(wav_4[1])-grad_4*Z[1]]


line_2_params,cov_2 = spo.curve_fit(line,Z,np.sqrt(1/wav_2),line_2_guess)
line_4_params,cov_4 = spo.curve_fit(line,Z,np.sqrt(1/wav_4),line_4_guess)
# print(grad_2**2)
# print(grad_4**2)

print(f"Grad squared N=2: {line_2_params[0]**2}")
print(f"Grad squared N=4: {line_4_params[0]**2}")

plt.scatter(Z,1/np.sqrt(wav_2))
plt.scatter(Z,1/np.sqrt(wav_4))
plt.plot(Z,line(Z,*line_2_params))
plt.plot(Z,line(Z,*line_4_params))
plt.show()