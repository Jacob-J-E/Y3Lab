import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import pandas as pd
import math
from scipy.signal import argrelextrema
import xraydb
R_0 = 10973731.6
H = 6.63e-34
C = 3e8
FS = 0.00729735

def correction_moseley(x,A,B,C):
    return A*((x-C)**2 + B*(x-C)**4)

def correction_moseley_quad(x,A,S):
    return A*((x-S)**2)

def correction_moseley_noB_alpha(x,A,S):
    B = (5/16)*(1/137)**2
    return A*((x-S)**2 + B*(x-S)**2)

def correction_moseley_noB_beta(x,A,S):
    B = (5/16)*(1/137)**2
    return A*((x-S)**2 + B*(x-S)**2)

# elements = ['Cu', 'Ag', 'Zr', 'Zn', 'Ni', 'Fe', 'Ti', 'Mo']
# Z = np.array([29,47,40,30,28,26,22,42])

elements = ['Ag', 'Zr','Mo']
Z = np.array([47,40,42])
wavelength_alpha = []
wavelength_beta = []
energy_alpha = []
energy_beta = []
for i in range(len(elements)):
    ka1 = 0
    kb1 = 0
    for name, line in xraydb.xray_lines(elements[i], 'K').items():
        if name == 'Ka1':
            ka1 = line.energy
        elif name == 'Kb1':
            kb1 = line.energy

    energy_alpha.append(ka1*1.6e-19)
    energy_beta.append(kb1*1.6e-19)

    wavelength_a = (H*C) / (ka1*1.6e-19)
    wavelength_b = (H*C) / (kb1*1.6e-19)

    wavelength_alpha.append(np.sqrt(1/wavelength_a))
    wavelength_beta.append(np.sqrt(1/wavelength_b))


A_alpha_guess = (3/4)*(H*C*R_0)
B_alpha_guess= (5/16)*(1/137)**2
grad_alpha_guess = (wavelength_alpha[1]-wavelength_alpha[0]) / (Z[1]-Z[0])
c_alpha_guess = wavelength_alpha[1] - grad_alpha_guess * Z[1]
alpha_guess = [A_alpha_guess,B_alpha_guess, 0]

A_beta_guess = (8/9)*(H*C*R_0)
B_beta_guess= (13/48)*(1/137)**2
grad_beta_guess = (wavelength_beta[1]-wavelength_beta[0]) / (Z[1]-Z[0])
c_beta_guess = wavelength_beta[1] - grad_beta_guess * Z[1]
beta_guess = [A_beta_guess,B_beta_guess, 0]

alpha_fit,alpha_cov = spo.curve_fit(correction_moseley,Z,energy_alpha,alpha_guess)
beta_fit,beta_cov = spo.curve_fit(correction_moseley,Z,energy_beta,beta_guess)

print(f'[ALPHA] | A: {alpha_fit[0]} A_theory: {A_alpha_guess} | B: {alpha_fit[1]} B_theory: {B_alpha_guess} | C (Screening): {alpha_fit[2]}|')
print(f'[BETA] | A: {beta_fit[0]} A_theory: {A_beta_guess} | B: {beta_fit[1]} B_theory: {B_beta_guess} | C (Screening): {beta_fit[2]}|')

print(f'[ALPHA] | R: {alpha_fit[0]/((3/4)*(H*C))} | fine structure: {np.sqrt((16/5)*alpha_fit[1])} ')
print(f'[BETA] | R: {beta_fit[0]/((8/9)*(H*C))} | fine structure: {np.sqrt((48/13)*beta_fit[1])} ')

print(f"Percentage Differnce alpha : {100*(R_0 - alpha_fit[0]/((3/4)*(H*C)))/R_0}")
print(f"Percentage Differnce beta : {100*(R_0 - beta_fit[0]/((8/9)*(H*C)))/R_0}")


print(f"alpha Percentage Differnce fine structure : {100*((FS) - np.sqrt((16/5)*alpha_fit[1]))/(FS)}")
print(f"beta Percentage Differnce fine structure : {100*((FS) - np.sqrt((48/13)*beta_fit[1]))/(FS)}")



sorted_z = sorted(Z)
plt.scatter(Z,energy_alpha,label = r'$k_/alpha$')
plt.plot(sorted_z,correction_moseley(sorted_z,*alpha_fit), label = r'Fitted $k_/alpha$ ')
plt.scatter(Z,energy_beta,label = r'$k_/beta$')
plt.plot(sorted_z,correction_moseley(sorted_z,*beta_fit), label = r'Fitted $k_/beta$')
plt.legend()
plt.show()

# A_alpha_guess = (3/4)*(H*C*R_0)
# grad_alpha_guess = (wavelength_alpha[1]-wavelength_alpha[0]) / (Z[1]-Z[0])
# c_alpha_guess = wavelength_alpha[1] - grad_alpha_guess * Z[1]
# alpha_guess = [A_alpha_guess, c_alpha_guess]

# A_beta_guess = (8/9)*(H*C*R_0)
# grad_beta_guess = (wavelength_beta[1]-wavelength_beta[0]) / (Z[1]-Z[0])
# c_beta_guess = wavelength_beta[1] - grad_beta_guess * Z[1]
# beta_guess = [A_beta_guess, c_beta_guess]

# alpha_fit,alpha_cov = spo.curve_fit(correction_moseley_quad,Z,energy_alpha,alpha_guess)
# beta_fit,beta_cov = spo.curve_fit(correction_moseley_quad,Z,energy_beta,beta_guess)

# print(f'[ALPHA] | A: {alpha_fit[0]} A_theory: {A_alpha_guess} | C (Screening): {alpha_fit[1]}|')
# print(f'[BETA] | A: {beta_fit[0]} A_theory: {A_beta_guess} | C (Screening): {beta_fit[1]}|')

# sorted_z = sorted(Z)
# plt.scatter(Z,energy_alpha,label = r'$k_/alpha$')
# plt.plot(sorted_z,correction_moseley_quad(sorted_z,*alpha_fit), label = r'Fitted $k_/alpha$ ')
# plt.scatter(Z,energy_beta,label = r'$k_/beta$')
# plt.plot(sorted_z,correction_moseley_quad(sorted_z,*beta_fit), label = r'Fitted $k_/beta$')
# plt.legend()
# plt.show()