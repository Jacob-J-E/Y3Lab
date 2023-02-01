import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import pandas as pd
import math
from scipy.signal import argrelextrema
import xraydb
R_0 = 10973731.6
H = 6.63e-34
c = 3e8

def straight_line(x,m,c):
    return m * x + c

elements = ['Cu', 'Ag', 'Zr', 'Zn', 'Ni', 'Fe', 'Ti', 'Mo']
Z = np.array([29,47,40,30,28,26,22,42])
wavelength_alpha = []
wavelength_beta = []
for i in range(len(elements)):
    ka1 = 0
    kb1 = 0
    for name, line in xraydb.xray_lines(elements[i], 'K').items():
        if name == 'Ka1':
            ka1 = line.energy
        elif name == 'Kb1':
            kb1 = line.energy

    wavelength_a = (H*c) / (ka1*1.6e-19)
    wavelength_b = (H*c) / (kb1*1.6e-19)

    wavelength_alpha.append(np.sqrt(1/wavelength_a))
    wavelength_beta.append(np.sqrt(1/wavelength_b))


grad_alpha_guess = (wavelength_alpha[1]-wavelength_alpha[0]) / (Z[1]-Z[0])
c_alpha_guess = wavelength_alpha[1] - grad_alpha_guess * Z[1]
alpha_guess = [grad_alpha_guess,c_alpha_guess]

grad_beta_guess = (wavelength_beta[1]-wavelength_beta[0]) / (Z[1]-Z[0])
c_beta_guess = wavelength_beta[1] - grad_beta_guess * Z[1]
beta_guess = [grad_beta_guess,c_beta_guess]

alpha_fit,alpha_cov = spo.curve_fit(straight_line,Z,wavelength_alpha,alpha_guess)
print(f'ALPHA LINE Gradient: {alpha_fit[0]}, calculated R: {(4/3)*alpha_fit[0]**2}')
beta_fit,beta_cov = spo.curve_fit(straight_line,Z,wavelength_beta,beta_guess)
print(f'ALPHA LINE Gradient: {beta_fit[0]}, calculated R: {(9/8)*beta_fit[0]**2}')

plt.scatter(Z,wavelength_alpha,label = r'$k_/alpha$')
plt.plot(Z,straight_line(Z,*alpha_fit), label = r'Fitted $k_/alpha$ ')
plt.scatter(Z,wavelength_beta,label = r'$k_/beta$')
plt.plot(Z,straight_line(Z,*beta_fit), label = r'Fitted $k_/beta$')
plt.legend()
plt.show()