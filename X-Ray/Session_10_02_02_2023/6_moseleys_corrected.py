import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import pandas as pd
import math
from scipy.signal import argrelextrema
import xraydb
import mplhep as hep
hep.style.use("ATLAS")
# from matplotlib import rc
# rc('text', usetex=True)
plt.style.use('dark_background')
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rcParams["text.usetex"]
R_0 = 10973731.6
H = 6.63e-34
C = 3e8
FS = 0.00729735

def chi_square(obs,exp):
    obs = np.array(obs)
    exp = np.array(exp)
    chi_val = (obs-exp)**2/exp
    return sum(chi_val)

def correction_moseley_alpha(x,A,C):
    return A*((x-C)**2 + (5/16)*(FS**2)*(x-C)**4)

def correction_moseley_beta(x,A,C):
    return A*((x-C)**2 + (13/48)*(FS**2)*(x-C)**4)

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

elements = ['Cu', 'Ag', 'Zr', 'Zn','Ti', 'Fe', 'Sn','Mo', 'In', 'Ni']
Z = np.array([29, 47, 40, 30, 22, 26, 50, 42, 49, 28])

# elements = ['Ag', 'Zr','Mo']
# Z = np.array([47,40,42])
wavelength_alpha = []
wavelength_beta = []
energy_alpha = []
energy_beta = []

energy_alpha = [8.080748431650838, 22.372552420613246, 16.044092033837604, 8.715570846388037, 4.57752299496841, 6.4190948273611665, 25.324555158862026, 17.421595970841373, 23.893491144556645, 7.681898645165174]
energy_beta = [8.95544201799072, 25.105444424220387, 17.982645794347498, 9.695802325558363, 5.097266130416366, 7.072766943625728, 28.42222963004282, 19.53560367390865, 26.785539971006223, 8.403630082046032]



energy_alpha = np.array(energy_alpha)*1e3*1.6e-19
energy_beta = np.array(energy_beta)*1e3*1.6e-19

y_error_k_a = [0.05*1e3*1.6e-19]*len(energy_alpha)
y_error_k_b = [0.05*1e3*1.6e-19]*len(energy_beta)

wavelength_a = (H*C) / (energy_alpha)
wavelength_b = (H*C) / (energy_beta)

wavelength_alpha = np.sqrt(1/wavelength_a)
wavelength_beta = np.sqrt(1/wavelength_b)


thres = 20
energy_alpha = energy_alpha[Z > thres]
energy_beta = energy_beta[Z > thres]
Z = Z[Z > thres]

A_alpha_guess = (3/4)*(H*C*R_0)
B_alpha_guess= (5/16)*(1/137)**2
grad_alpha_guess = (wavelength_alpha[1]-wavelength_alpha[0]) / (Z[1]-Z[0])
c_alpha_guess = wavelength_alpha[1] - grad_alpha_guess * Z[1]
#alpha_guess = [A_alpha_guess,B_alpha_guess, 0]
alpha_guess = [A_alpha_guess, 0]

A_beta_guess = (8/9)*(H*C*R_0)
B_beta_guess= (13/48)*(1/137)**2
grad_beta_guess = (wavelength_beta[1]-wavelength_beta[0]) / (Z[1]-Z[0])
c_beta_guess = wavelength_beta[1] - grad_beta_guess * Z[1]
#beta_guess = [A_beta_guess,B_beta_guess, 0]
beta_guess = [A_beta_guess, 0]


abol = True
alpha_fit,alpha_cov = spo.curve_fit(correction_moseley_alpha,Z,energy_alpha,alpha_guess, sigma = y_error_k_a, absolute_sigma= abol)
beta_fit,beta_cov = spo.curve_fit(correction_moseley_beta,Z,energy_beta,beta_guess, sigma = y_error_k_b, absolute_sigma= abol)

# print(f'[ALPHA] | A: {alpha_fit[0]} A_theory: {A_alpha_guess} | B: {alpha_fit[1]} B_theory: {B_alpha_guess} | C (Screening): {alpha_fit[2]}|')
# print(f'[BETA] | A: {beta_fit[0]} A_theory: {A_beta_guess} | B: {beta_fit[1]} B_theory: {B_beta_guess} | C (Screening): {beta_fit[2]}|')

# print(f'[ALPHA] | R: {alpha_fit[0]/((3/4)*(H*C))} | fine structure: {np.sqrt((16/5)*alpha_fit[1])} ')
# print(f'[BETA] | R: {beta_fit[0]/((8/9)*(H*C))} | fine structure: {np.sqrt((48/13)*beta_fit[1])} ')

# print(f"Percentage Differnce alpha : {100*(R_0 - alpha_fit[0]/((3/4)*(H*C)))/R_0}")
# print(f"Percentage Differnce beta : {100*(R_0 - beta_fit[0]/((8/9)*(H*C)))/R_0}")


# print(f"alpha Percentage Differnce fine structure : {100*((FS) - np.sqrt((16/5)*alpha_fit[1]))/(FS)}")
# print(f"beta Percentage Differnce fine structure : {100*((FS) - np.sqrt((48/13)*beta_fit[1]))/(FS)}")

print(f'[ALPHA] | A: {alpha_fit[0]} A_theory: {A_alpha_guess}| C (Screening): {alpha_fit[1]}|')
print(f'[BETA] | A: {beta_fit[0]} A_theory: {A_beta_guess}| C (Screening): {beta_fit[1]}|')

print(f'[ALPHA] | R: {alpha_fit[0]/((3/4)*(H*C))}')
print(f'[BETA] | R: {beta_fit[0]/((8/9)*(H*C))}')


def chi_square(obs,exp):
    obs = np.array(obs)
    exp = np.array(exp)
    chi_val = (obs-exp)**2/exp**2
    return sum(chi_val)


k_alpha_rydberg = alpha_fit[0]/((3/4)*(H*C))
k_beta_rydberg = beta_fit[0]/((8/9)*(H*C))

def cal_rydberg(m,factor):
    return (1/(factor*(H*C)))*m


sorted_z,energy_alpha_sorted,energy_beta_sorted = zip(*sorted(zip(Z, energy_alpha,energy_beta)))
energy_alpha_sorted = np.array(energy_alpha_sorted)
energy_beta_sorted = np.array(energy_beta_sorted)
sorted_z = np.array(sorted_z)

print(f"Percentage Differnce alpha : {100*(R_0 - k_alpha_rydberg)/R_0}")
print(f"Percentage Differnce beta : {100*(R_0 - k_beta_rydberg)/R_0}")

print(f"Percentage Differnce k_alpha: {(100*(R_0 - k_alpha_rydberg)/R_0):.4g} with value {k_alpha_rydberg:.4g} with error, {cal_rydberg(np.sqrt(alpha_cov[0][0]),3/4):.4g}, chi - {chi_square(energy_alpha_sorted,correction_moseley_alpha(sorted_z,*alpha_fit)):.4g}")
print(f"Percentage Differnce k_beta: {(100*(R_0 - k_beta_rydberg)/R_0):.4g} with value {k_beta_rydberg:.4g} with error, {cal_rydberg(np.sqrt(beta_cov[0][0]),8/9):.4g}, chi - {chi_square(energy_beta_sorted,correction_moseley_beta(sorted_z,*beta_fit)):.4g}")



sigma_E = 0.05 
plot_sigma = np.zeros_like(energy_alpha) + 0.05
keV = 1e3*1.6e-19


k_domain_invalid = np.arange(min(sorted_z),31,1)
k_domain_valid = np.arange(30,max(sorted_z)+1,1)


plt.scatter(Z,energy_alpha/keV,label = r'$k_/alpha$',color="#46bddf",marker='x')
#plt.plot(sorted_z,correction_moseley_alpha(sorted_z,*alpha_fit)/keV, label = r'Fitted $k_/alpha$',color="#46bddf")
plt.plot(k_domain_valid,correction_moseley_alpha(k_domain_valid,*alpha_fit)/keV, label = r'Fitted $k_/alpha$',color="#46bddf")
plt.plot(k_domain_invalid,correction_moseley_alpha(k_domain_invalid,*alpha_fit)/keV, linestyle = '--',color="#46bddf")
plt.scatter(Z,energy_beta/keV,label = r'$k_/beta$',color="#f05464",marker='x')
#plt.plot(sorted_z,correction_moseley_beta(sorted_z,*beta_fit)/keV, label = r'Fitted $k_/beta$',color="#f05464")
plt.plot(k_domain_valid,correction_moseley_alpha(k_domain_valid,*beta_fit)/keV, label = r'Fitted $k_/beta$',color="#f05464")
plt.plot(k_domain_invalid,correction_moseley_alpha(k_domain_invalid,*beta_fit)/keV, linestyle = '--',color="#f05464")

atomic_numbers_ordered = [21,26,28,29,30,40,42,47,49,50] 
atomic_name_ordered = ['Ti','Fe','Ni','Cu','Zn','Zr','Mo','Ag','In','Sn'] 

for i,name in enumerate(atomic_name_ordered):
    plt.text(atomic_numbers_ordered[i] - 0.5, energy_beta_sorted[i]/keV, name, horizontalalignment='right', size='medium', color='white', weight='semibold')

plt.text(atomic_numbers_ordered[-1]+1, energy_alpha[-4]/keV, r'$K_{\alpha}$', horizontalalignment='left', size='large', color=u"#46bddf", weight='bold')
plt.text(atomic_numbers_ordered[-1]+1, energy_beta[-4]/keV, r'$K_{\beta}$', horizontalalignment='left', size='large', color=u"#f05464", weight='bold')
# plt.text()
# plt.legend()
plt.xlim(17,55)
plt.ylim(2,35)
plt.xlabel("Atomic Number")
plt.ylabel(r"Energy (keV)")
plt.grid(alpha=0.5)
plt.savefig(r"X-Ray\Plots and Figs\correct_moseley.png",dpi=400,format="png")
plt.show()

# plt.plot(Z_plot,moseley,label="Moseley's Law",color="#46bddf")
# plt.plot(Z_plot,K_alpha,label=r"$K_{\alpha} Line$",color="#e4e5e7")
# plt.plot(Z_plot,K_beta,label=r"$K_{\beta} Line$",color="#f05464")

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