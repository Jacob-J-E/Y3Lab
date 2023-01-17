import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("ATLAS")


data = pd.read_csv(r"X-Ray\Data\16-01-2022\NaCl Full Data.csv",skiprows=0)
print(data)


angle = data['angle']
wav = data['wav / pm']
energy = data['E / keV']
count_0 = data['R_0 / 1/s']

A = 564.02e-12
ENERGY1 = 17.443e3*1.6e-19
ENERGY2 = 19.651e3*1.6e-19

def calculate_angle(h,k,l,energy):
    wavelength = (6.63e-34 * 3e8)/energy
    sin_angle = np.sqrt((wavelength**2/(4*A**2))*(h**2+k**2+l**2))
    angle = np.arcsin(sin_angle)*180/np.pi
    return angle

primitive =[]
bcc = []
fcc = []
hcp = []

def bcc_check(h,k,l):
    if (h+k+l)%2 == 0:
        return True
    return False

def fcc_check(h,k,l):
    if (h%2 == 0 and k%2 == 0 and l%2 == 0) or(h%2 == 1 and k%2 == 1 and l%2 == 1):
        return True
    return False

def hcp_check()

range_max = range(0,10)
for h in range_max:
    for k in range_max:
        for l in range_max:
            print(h,k,l)