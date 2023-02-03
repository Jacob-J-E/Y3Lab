import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("ATLAS")

# Bragg scattering error propagation function
def sigma_wav(m,d,theta,sig_theta,sig_d):
    der_d = 2 * np.sin(theta) / m
    der_theta = 2 * d * np.cos(theta) / m
    return np.sqrt((der_d * sig_d)**2 + (der_theta * sig_theta))


# Load in data
data = pd.read_csv(r"X-Ray\Data\16-01-2022\NaCl Full Data.csv",skiprows=0)
print(data)

# Split data into differnt variables
angle = data['angle']
wav = data['wav / pm']
energy = data['E / keV']
count_0 = data['R_0 / 1/s']
