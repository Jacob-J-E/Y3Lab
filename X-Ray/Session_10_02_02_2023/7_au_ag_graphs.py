import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import pandas as pd
import math
from scipy.signal import argrelextrema
import xraydb
import pickle


#Loading in old low res data
data_low_res = pd.read_csv(r"X-Ray\Data\24-01-2023\Plates Mo Source.csv",skiprows=0)

#Loading in old high res data
data_high_res = pickle.load(open(r'X-Ray\Data\02-02-2023\MEGA DATA.pkl', 'rb'))


def gaussian(x, a, b, c,e,f,g):
    return (a * np.exp(-((x - b) ** 2) / (2 * c ** 2)) + e * np.exp(-((x - f) ** 2) / (2 * g ** 2)))

def gaussian_triple(x, a, b, c,e,f,g,h,i,j):
    return (a * np.exp(-((x - b) ** 2) / (2 * c ** 2)) + e * np.exp(-((x - f) ** 2) / (2 * g ** 2)) + h * np.exp(-((x - i) ** 2) / (2 * j ** 2)))

ag = 'Ag Plate'
au = 'Gold (Ganel’s pendant)'

ag_plate = data_low_res['Ag Plate']
energy_low_res_x = data_low_res['E_1 / keV']
energy_high_res_x = data_high_res['E_1 / keV']


ka1 = 0
kb1 = 0
for name, line in xraydb.xray_lines('Ag', 'K').items():
    if name == 'Ka1':
        ka1 = line.energy
    elif name == 'Kb1':
        kb1 = line.energy

mu_guess_e1 = ka1/1e3
mu_guess_e2 = kb1/1e3


ag_plate_np = np.array(ag_plate.tolist())
energy_np = np.array(energy_low_res_x.tolist())
# Find local maxima using the argrelextrema function
local_maxima = argrelextrema(ag_plate.to_numpy(), np.greater)

amplitudes = []
for x in local_maxima[0]:
    amplitudes.append(ag_plate_np[x])

not_sorted_amplitudes = amplitudes.copy()
amplitudes.sort(reverse=True)

guess_e1 = [amplitudes[0],mu_guess_e1,0.1,amplitudes[1],mu_guess_e2,0.1]
params_e1, cov_e1 = spo.curve_fit(gaussian,energy_np,ag_plate_np,guess_e1)

if params_e1[1] < params_e1[4]:
    alpha_energies_ag = params_e1[1]
    beta_energies_ag = params_e1[4]
else:
    alpha_energies_ag = params_e1[4]
    beta_energies_ag = params_e1[1]





plt.figure(1)
plt.plot(energy_low_res_x,ag_plate, label = f'Ag Plate') 
plt.plot(energy_low_res_x,gaussian(energy_low_res_x,*params_e1), label = f'Gaussian fit (mu_e1 = {params_e1[1]:.2f},mu_e2 = {params_e1[4]:.2f})')
plt.axvline(mu_guess_e1, label = r'$k_{\alpha}$' + f' with theoretical energy {mu_guess_e1:.2f} keV', color = 'blue')
plt.axvline(mu_guess_e2, label = r'$k_{\beta}$' + f' with theoretical energy {mu_guess_e2:.2f} keV', color = 'red')
plt.xlim(left = 2, right = 35)
plt.legend(loc="upper right")

au_plate = data_high_res[au]


la1 = 0
lb1 = 0
lg1 = 0
for name, line in xraydb.xray_lines('Au').items():
    if name == 'La1':
        la1 = line.energy
    elif name == 'Lb1':
        lb1 = line.energy
    elif name == 'Lg1':
        lg1 = line.energy

mu_guess_e1 = la1/1e3
mu_guess_e2 = lb1/1e3
mu_guess_e3 = lg1/1e3


au_plate_np = np.array(au_plate.tolist())
energy_np = np.array(energy_high_res_x.tolist())
# Find local maxima using the argrelextrema function
local_maxima = argrelextrema(au_plate.to_numpy(), np.greater)

amplitudes = []
for x in local_maxima[0]:
    amplitudes.append(au_plate_np[x])

not_sorted_amplitudes = amplitudes.copy()
amplitudes.sort(reverse=True)

guess_e1 = [amplitudes[0],mu_guess_e1,0.1,amplitudes[1],mu_guess_e2,0.1,amplitudes[2],mu_guess_e3,0.1]
params_e1, cov_e1 = spo.curve_fit(gaussian_triple,energy_np,au_plate_np,guess_e1)

energy_val = [params_e1[1],params_e1[4],params_e1[7]]
energy_val = sorted(energy_val)

l_alpha_energies = energy_val[2]
l_beta_energies = energy_val[1]
l_gamma_energies = energy_val[0]

plt.figure(2)
plt.plot(energy_high_res_x,au_plate, label = f'Au Plate') 
plt.plot(energy_high_res_x,gaussian_triple(energy_high_res_x,*params_e1), label = f'Gaussian fit (mu_e1 = {params_e1[1]:.2f},mu_e2 = {params_e1[4]:.2f},mu_e3 = {params_e1[7]:.2f})')
plt.axvline(mu_guess_e1, label = r'$l_{\alpha}$' + f' with energy {mu_guess_e1:.2f} keV', color = 'blue')
plt.axvline(mu_guess_e2, label = r'$l_{\beta}$' + f' with energy {mu_guess_e2:.2f} keV', color = 'red')
plt.axvline(mu_guess_e3, label = r'$l_{\gamma}$' + f' with energy {mu_guess_e3:.2f} keV', color = 'green')
plt.legend(loc="upper right")
plt.xlim(left = 2, right = 35)
plt.show()