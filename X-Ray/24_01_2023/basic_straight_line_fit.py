import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

Z = np.array([47,42,40])
wavelength = np.array([47.2,60.4,66.8])*1e-12
inverse_sqrt_wave = 1/np.sqrt(wavelength)

def line(x,m,c):
    return m*x + c

R_0 = 10973731.6

alpha_guess = [np.sqrt(R_0),0]

alpha_params, alpha_cov = spo.curve_fit(line,Z,inverse_sqrt_wave,alpha_guess)


print(alpha_params**2)