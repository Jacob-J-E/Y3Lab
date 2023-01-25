import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import pandas as pd
import math
from scipy.signal import argrelextrema
import xraydb



def gaussian(x, a, b, c,e,f,g):
    return (a * np.exp(-((x - b) ** 2) / (2 * c ** 2)) + e * np.exp(-((x - f) ** 2) / (2 * g ** 2)))

first_run = pd.read_csv(r"X-Ray\Data\24-01-2023\Plates Mo Source.csv",skiprows=0)

columns = first_run.columns.tolist()

columns.remove('Mo Source')
columns.remove('FeCr Alloy Plate')
columns.remove('Tin Maybe Plate')
columns.remove('Cu Maybe')
columns.remove('Tin Maybe Plate.1')
columns.remove('E_1 / keV')

print(columns)
print("Total number of graphs",len(columns))
Tot = len(columns)
Cols = 4
Rows = Tot // Cols 
if Tot % Cols != 0:
    Rows += 1
Position = range(1,Tot + 1)


elements = ['Cu', 'Ag', 'Zr', 'Zn', 'Ni', 'Fe', 'Ti', 'Mo']
print(first_run.head())
fig = plt.figure(1)
for i,col_name in enumerate(columns):
    energy = first_run['E_1 / keV']
    cu_plate = first_run[col_name]


    ka1 = 0
    kb1 = 0
    for name, line in xraydb.xray_lines(elements[i], 'K').items():
        if name == 'Ka1':
            ka1 = line.energy
        elif name == 'Kb1':
            kb1 = line.energy

    mu_guess_e1 = ka1/1e3
    mu_guess_e2 = kb1/1e3

    print(mu_guess_e2)





    cu_plate_np = np.array(cu_plate.tolist())
    energy_np = np.array(energy.tolist())
    # Find local maxima using the argrelextrema function
    local_maxima = argrelextrema(cu_plate.to_numpy(), np.greater)

    amplitudes = []
    for x in local_maxima[0]:
        amplitudes.append(cu_plate_np[x])

    amplitudes.sort(reverse=True)


    guess_e1 = [amplitudes[0],mu_guess_e1,0.1,amplitudes[1],mu_guess_e2,0.1]
    params_e1, cov_e1 = spo.curve_fit(gaussian,energy_np,cu_plate_np,guess_e1)


    print(f"mu_e1: {params_e1[1]}")
    print(f"sigma_e1: {params_e1[2]}")
    print(f"mu_e2: {params_e1[4]}")
    print(f"sigma_e2: {params_e1[5]}")

    print(f"cov mu_e1: {np.sqrt(cov_e1[0][0])}")
    print(f"covsigma_e1: {np.sqrt(cov_e1[1][1])}")
    print(f"cov mu_e2: {np.sqrt(cov_e1[2][2])}")
    print(f"covsigma_e2: {np.sqrt(cov_e1[3][3])}")

    # plt.plot(energy,cu_plate, label = f'{elements[i]} Plate')
    # plt.plot(energy,gaussian(energy,*params_e1), label = f'Gaussian fit (mu_e1 = {params_e1[1]},mu_e2 = {params_e1[4]})')
    # plt.legend()
    # plt.show()
    ax = fig.add_subplot(Rows,Cols,Position[i])
    ax.plot(energy,cu_plate, label = f'{elements[i]} Plate') 
    ax.plot(energy,gaussian(energy,*params_e1), label = f'Gaussian fit (mu_e1 = {params_e1[1]:.2f},mu_e2 = {params_e1[4]:.2f})')
    ax.legend(loc="upper right")
plt.show()