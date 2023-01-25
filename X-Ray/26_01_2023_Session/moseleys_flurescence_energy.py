import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import pandas as pd
import math
from scipy.signal import argrelextrema



def gaussian(x, a, b, c):
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))

first_run = pd.read_csv(r"X-Ray\Data\24-01-2023\Plates Mo Source.csv",skiprows=0)


#print(first_run.head())


energy = first_run['E_1 / keV']
cu_plate = first_run['Cu Plate']

mu_guess = 8.04





cu_plate_np = np.array(cu_plate.tolist())
energy_np = np.array(energy.tolist())
# Find local maxima using the argrelextrema function
local_maxima = argrelextrema(cu_plate.to_numpy(), np.greater)

amplitudes = []
for x in local_maxima[0]:
    amplitudes.append(cu_plate_np[x])

amplitudes.sort(reverse=True)


guess_e1 = [amplitudes[0],mu_guess,0.1]
params_e1, cov_e1 = spo.curve_fit(gaussian,energy_np,cu_plate_np,guess_e1)


print(f"mu_e1: {params_e1[0]}")
print(f"sigma_e1: {params_e1[1]}")

print(f"cov mu_e1: {np.sqrt(cov_e1[0][0])}")
print(f"covsigma_e1: {np.sqrt(cov_e1[1][1])}")

guess_e2 = [amplitudes[1],mu_guess,0.1]
params_e2, cov_e2 = spo.curve_fit(gaussian,energy_np,cu_plate_np,guess_e2)


print(f"mu_e2: {params_e2[0]}")
print(f"sigma_e2: {params_e2[1]}")

print(f"cov mu_e2: {np.sqrt(cov_e2[0][0])}")
print(f"covsigma_e2: {np.sqrt(cov_e2[1][1])}")

plt.plot(energy,cu_plate, label = 'Cu Plate')
plt.plot(energy,gaussian(energy,*params_e1), label = 'Gaussian fit E1')
plt.plot(energy,gaussian(energy,*params_e2), label = 'Gaussian fit E2')
plt.legend()
plt.show()
