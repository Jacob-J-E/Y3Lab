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
local_maxima = argrelextrema(cu_plate_np, np.greater)

print(cu_plate[local_maxima[0]][0])
print(cu_plate[local_maxima[0]][1])

guess = [100,mu_guess,0.1]
params, cov = spo.curve_fit(gaussian,energy_np,cu_plate_np,guess)


print(f"mu: {params[0]}")
print(f"sigma: {params[1]}")

print(f"cov mu: {np.sqrt(cov[0][0])}")
print(f"covsigma: {np.sqrt(cov[1][1])}")

plt.plot(energy,cu_plate, label = 'Cu Plate')
plt.plot(energy,gaussian(energy,*params), label = 'Gaussian fit')
plt.legend()
plt.show()
