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

#all the column names
column_names_data_low_res_inital = ['E_1 / keV', 'Cu Plate', 'Mo Source', 'Ag Plate', 'Zr Plate',
       'Zn Plate', 'Ni Plate', 'Fe Plate', 'FeCr Alloy Plate',
       'Tin Maybe Plate', 'Titanium Plate', 'Cu Maybe', 'Tin Maybe Plate.1',
       'Mo Plate']



k_line_low_res_search = ['Cu Plate', 'Ag Plate', 'Zr Plate',
       'Zn Plate','Titanium Plate', 'Fe Plate', 'Tin Maybe Plate.1']

k_elements_low_res = ['Cu', 'Ag', 'Zr', 'Zn','Ti', 'Fe', 'Sn']
low_res_atomic_number = [29, 47, 40, 30]

energy_low_res_x = data_low_res['E_1 / keV']


#Loading in old high res data
data_high_res = pickle.load(open(r'X-Ray\Data\02-02-2023\MEGA DATA.pkl', 'rb'))

#all the column names
column_names_data_high_res = ['E_1 / keV', 'Mo plate cal', 'Cu straight through calibrate',
       'Unknown - Tungsten (W)', 'Unknown - Ti', 'Pb', ' In +  plastic',
       'Tin/Copper', 'Fe Alloy', 'Iron Zinc', 'Pure Iron',
       'Gold (Ganel’s pendant)', 'Iron Nickel', 'Gold (Ganel’s chain)',
       'Nickel (Tulsi’s Ring)', 'inconclusive (Ben’s Ring)',
       'Manganin (Cu Mn Ni)', 'Iron (with possible mixing)',
       'Stainless steel ', 'Solder (tin)', 'Nickel-Brass (Ganel’s Key)',
       '50 Florint ', '10 peso (Copper Zinc)', 'Israel (Copper Zinc Nickel)',
       'HK Dollar', '5 Euro Cent (primarily copper)', 'Indium', 'Pure Iron.1',
       'Pure Nickel']

energy_high_res_x = data_high_res['E_1 / keV']

#These are the other elements we think we have identified. (Ignoring alloys for this section)
column_names_data_high_res_search = ['Mo plate cal',
       'Unknown - Tungsten (W)', 'Unknown - Ti', 'Pb', 'Pure Iron', 'Gold (Ganel’s pendant)', 'Solder (tin)','Indium',
       'Pure Nickel']

k_line_high_res_search = ['Mo plate cal','Indium','Pure Nickel']
k_elements_high_res = ['Mo', 'In', 'Ni']

l_line_high_res_search = ['Unknown - Tungsten (W)', 'Pb' 'Gold (Ganel’s pendant)']
l_elements_high_res = ['W', 'Pb', 'Au']

high_res_atomic_number = [42,74,22,82, 26, 79, 50,49, 28]

def gaussian(x, a, b, c,e,f,g):
    return (a * np.exp(-((x - b) ** 2) / (2 * c ** 2)) + e * np.exp(-((x - f) ** 2) / (2 * g ** 2)))


plot = True
if plot == True:
    Tot = len(k_line_low_res_search+k_line_high_res_search)
    Cols = 4
    Rows = Tot // Cols 
    if Tot % Cols != 0:
        Rows += 1
    Position = range(1,Tot + 1)
    fig = plt.figure(1)

alpha_energies = []
beta_energies = []
j = 0
for i,col_name in enumerate(k_line_low_res_search):
    cu_plate = data_low_res[col_name]


    ka1 = 0
    kb1 = 0
    for name, line in xraydb.xray_lines(k_elements_low_res[i], 'K').items():
        if name == 'Ka1':
            ka1 = line.energy
        elif name == 'Kb1':
            kb1 = line.energy

    mu_guess_e1 = ka1/1e3
    mu_guess_e2 = kb1/1e3

    print(mu_guess_e2)

    cu_plate_np = np.array(cu_plate.tolist())
    energy_np = np.array(energy_low_res_x.tolist())
    # Find local maxima using the argrelextrema function
    local_maxima = argrelextrema(cu_plate.to_numpy(), np.greater)

    amplitudes = []
    for x in local_maxima[0]:
        amplitudes.append(cu_plate_np[x])

    not_sorted_amplitudes = amplitudes.copy()
    amplitudes.sort(reverse=True)

    guess_e1 = [amplitudes[0],mu_guess_e1,0.1,amplitudes[1],mu_guess_e2,0.1]
    params_e1, cov_e1 = spo.curve_fit(gaussian,energy_np,cu_plate_np,guess_e1)
    if params_e1[1] < params_e1[4]:
        alpha_energies.append(params_e1[1])
        beta_energies.append(params_e1[4])
    else:
        alpha_energies.append(params_e1[4])
        beta_energies.append(params_e1[1])


    print(f"mu_e1: {params_e1[1]}")
    print(f"sigma_e1: {params_e1[2]}")
    print(f"mu_e2: {params_e1[4]}")
    print(f"sigma_e2: {params_e1[5]}")

    print(f"cov mu_e1: {np.sqrt(cov_e1[0][0])}")
    print(f"covsigma_e1: {np.sqrt(cov_e1[1][1])}")
    print(f"cov mu_e2: {np.sqrt(cov_e1[2][2])}")
    print(f"covsigma_e2: {np.sqrt(cov_e1[3][3])}")

    if plot == True:
        ax = fig.add_subplot(Rows,Cols,Position[j])
        ax.plot(energy_low_res_x,cu_plate, label = f'{k_elements_low_res[i]} Plate') 
        ax.plot(energy_low_res_x,gaussian(energy_low_res_x,*params_e1), label = f'Gaussian fit (mu_e1 = {params_e1[1]:.2f},mu_e2 = {params_e1[4]:.2f})')
        plt.axvline(mu_guess_e1, label = r'$k_{\alpha}$' + f' with energy {mu_guess_e1:.2f} keV', color = 'blue')
        plt.axvline(mu_guess_e2, label = r'$k_{\alpha}$' + f' with energy {mu_guess_e2:.2f} keV', color = 'red')
        ax.legend(loc="upper right")
    j += 1

for i,col_name in enumerate(k_line_high_res_search):
    cu_plate = data_high_res[col_name]


    ka1 = 0
    kb1 = 0
    for name, line in xraydb.xray_lines(k_elements_high_res[i], 'K').items():
        if name == 'Ka1':
            ka1 = line.energy
        elif name == 'Kb1':
            kb1 = line.energy

    mu_guess_e1 = ka1/1e3
    mu_guess_e2 = kb1/1e3

    print(mu_guess_e2)

    cu_plate_np = np.array(cu_plate.tolist())
    energy_np = np.array(energy_high_res_x.tolist())
    # Find local maxima using the argrelextrema function
    local_maxima = argrelextrema(cu_plate.to_numpy(), np.greater)

    amplitudes = []
    for x in local_maxima[0]:
        amplitudes.append(cu_plate_np[x])

    not_sorted_amplitudes = amplitudes.copy()
    amplitudes.sort(reverse=True)

    guess_e1 = [amplitudes[0],mu_guess_e1,0.1,amplitudes[1],mu_guess_e2,0.1]
    params_e1, cov_e1 = spo.curve_fit(gaussian,energy_np,cu_plate_np,guess_e1)
    if params_e1[1] < params_e1[4]:
        alpha_energies.append(params_e1[1])
        beta_energies.append(params_e1[4])
    else:
        alpha_energies.append(params_e1[4])
        beta_energies.append(params_e1[1])


    print(f"mu_e1: {params_e1[1]}")
    print(f"sigma_e1: {params_e1[2]}")
    print(f"mu_e2: {params_e1[4]}")
    print(f"sigma_e2: {params_e1[5]}")

    print(f"cov mu_e1: {np.sqrt(cov_e1[0][0])}")
    print(f"covsigma_e1: {np.sqrt(cov_e1[1][1])}")
    print(f"cov mu_e2: {np.sqrt(cov_e1[2][2])}")
    print(f"covsigma_e2: {np.sqrt(cov_e1[3][3])}")

    if plot == True:
        ax = fig.add_subplot(Rows,Cols,Position[j])
        ax.plot(energy_high_res_x,cu_plate, label = f'{k_elements_high_res[i]} Plate') 
        ax.plot(energy_high_res_x,gaussian(energy_high_res_x,*params_e1), label = f'Gaussian fit (mu_e1 = {params_e1[1]:.2f},mu_e2 = {params_e1[4]:.2f})')
        plt.axvline(mu_guess_e1, label = r'$k_{\alpha}$' + f' with energy {mu_guess_e1:.2f} keV', color = 'blue')
        plt.axvline(mu_guess_e2, label = r'$k_{\alpha}$' + f' with energy {mu_guess_e2:.2f} keV', color = 'red')
        ax.legend(loc="upper right")
    j += 1

plt.show()