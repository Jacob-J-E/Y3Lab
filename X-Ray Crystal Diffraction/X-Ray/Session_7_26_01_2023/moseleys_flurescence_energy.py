import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import pandas as pd
import math
from scipy.signal import argrelextrema
import xraydb
import mplhep as hep
from scipy.signal import argrelextrema
# hep.style.use("ATLAS")
# plt.style.use('dark_background')
# plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
# plt.rcParams["text.usetex"]
R_0 = 10973731.6

def straight_line(x,m,c):
    return m * x + c



def gaussian(x, a, b, c,e,f,g):
    return (a * np.exp(-((x - b) ** 2) / (2 * c ** 2)) + e * np.exp(-((x - f) ** 2) / (2 * g ** 2)))

first_run = pd.read_csv(r"X-Ray\Data\24-01-2023\Plates Mo Source.csv",skiprows=0)

columns = first_run.columns.tolist()

columns.remove('Mo Source')
columns.remove('FeCr Alloy Plate')
#columns.remove('Tin Maybe Plate')
columns.remove('Cu Maybe')
#columns.remove('Tin Maybe Plate.1')
columns.remove('E_1 / keV')


elements = ['Cu', 'Ag', 'Zr', 'Zn', 'Ni', 'Fe', 'Sn', 'Ti', 'Sn', 'Mo']
Z = np.array([29,47,40,30,28,26,50,22,50,42])
A = np.array([63.5,107.87,91.224,65.38,58.693,55.845,118.71,47.867,118.71,95.95])
print(first_run.head())

plot = True
if plot == True:
    Tot = len(columns)
    Cols = 4
    Rows = Tot // Cols 
    if Tot % Cols != 0:
        Rows += 1
    Position = range(1,Tot + 1)
    fig = plt.figure(1)

alpha_energies = []
beta_energies = []

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
    # energy_1 = energy[int(cu_plate.tolist().index(amplitudes[0]))]
    # energy_2 = energy[int(cu_plate.tolist().index(amplitudes[1]))]
    # if elements[i] == 'Fe':
    #     alpha_energies.append(energy_1)
    #     beta_energies.append(7.00)
    # else:
    #     if energy_1 > energy_2:
    #         alpha_energies.append(energy_1)
    #         beta_energies.append(energy_2)
    #     else:
    #         alpha_energies.append(energy_2)
    #         beta_energies.append(energy_1)

    # print(f'energy 1 : {energy_1}')
    # print(f'energy 2 : {energy_2}')
    if plot == True:
        ax = fig.add_subplot(Rows,Cols,Position[i])
        ax.plot(energy,cu_plate, label = f'{elements[i]} Plate') 
        ax.plot(energy,gaussian(energy,*params_e1), label = f'Gaussian fit (mu_e1 = {params_e1[1]:.2f},mu_e2 = {params_e1[4]:.2f})')
        #ax.scatter([energy_1,energy_2],[amplitudes[0],amplitudes[1]],label = 'max peaks')
        ax.legend(loc="upper right")


alpha_plot = np.sqrt(np.array(alpha_energies)*1e3*1.6e-19 / (6.63e-34 * 3e8))
beta_plot = np.sqrt(np.array(beta_energies)*1e3*1.6e-19 / (6.63e-34 * 3e8))
if plot == True:
    fig,ax = plt.subplots(1,2,figsize=(6,6))
    ax[0].scatter(Z,alpha_plot,color='black',marker='x',label=r'$K_ \alpha$')
    ax[0].scatter(Z,beta_plot,color='grey',marker='x',label=r'$K_{\beta}$')
    ax[0].set_xlabel("Atomic Number (Z)")
    ax[0].set_ylabel(r"$\sqrt{1/\lambda}$")
    ax[0].grid()

    ax[1].scatter(A,alpha_plot,color='black',marker='x',label='Alpha Lines')
    ax[1].scatter(A,beta_plot,color='grey',marker='x',label="Beta Lines")
    ax[1].set_xlabel("Atomic Weight (A)")
    ax[1].set_ylabel(r"$\sqrt{1/\lambda}$")
    ax[1].grid()


    Z_30 = Z[Z<40]
    Z_40 = Z[Z > 35]
    alpha_30 = alpha_plot[Z<40]
    beta_30 = beta_plot[Z<40]

    alpha_40 = alpha_plot[Z > 35]
    beta_40 = beta_plot[Z > 35]

    grad_alpha_40_guess = (alpha_40[1]-alpha_40[0]) / (Z_40[1]-Z_40[0])
    c_alpha_40_guess = alpha_40[1] - grad_alpha_40_guess * Z_40[1]
    alpha_40_guess = [grad_alpha_40_guess,c_alpha_40_guess]

    grad_beta_40_guess = (beta_40[1]-beta_40[0]) / (Z_40[1]-Z_40[0])
    c_beta_40_guess = beta_40[1] - grad_beta_40_guess * Z_40[1]
    beta_40_guess = [grad_beta_40_guess,c_beta_40_guess]

    alpha_40_fit,alpha_40_cov = spo.curve_fit(straight_line,Z_40,alpha_40,alpha_40_guess)
    beta_40_fit,beta_40_cov = spo.curve_fit(straight_line,Z_40,beta_40,beta_40_guess)

    grad_alpha_30_guess = (alpha_30[1]-alpha_30[0]) / (Z_30[1]-Z_30[0])
    c_alpha_30_guess = alpha_30[1] - grad_alpha_30_guess * Z_30[1]
    alpha_30_guess = [grad_alpha_30_guess,c_alpha_30_guess]

    grad_beta_30_guess = (beta_30[1]-beta_30[0]) / (Z_30[1]-Z_30[0])
    c_beta_30_guess = beta_30[1] - grad_beta_30_guess * Z_30[1]
    beta_30_guess = [grad_beta_30_guess,c_beta_30_guess]

    alpha_30_fit,alpha_30_cov = spo.curve_fit(straight_line,Z_30,alpha_30,alpha_30_guess)
    beta_30_fit,beta_30_cov = spo.curve_fit(straight_line,Z_30,beta_30,beta_30_guess)

    ax[0].plot(Z_40,straight_line(Z_40,*alpha_40_fit),color='green',label=r"Z>40 $K_{\alpha}$ Fit line")
    ax[0].plot(Z_40,straight_line(Z_40,*beta_40_fit),color='blue',label=r"Z>40 $K_{\beta}$ Fit line")

    ax[0].plot(Z_30,straight_line(Z_30,*alpha_30_fit),color='red',label=r"Z<40 $K_{\alpha}$ Fit line",linestyle='-')
    ax[0].plot(Z_30,straight_line(Z_30,*beta_30_fit),color='orange',label=r"Z<40 $K_{\alpha}$ Fit line",linestyle='-')

    ax[0].set_title("Atomic number against inverse root wavelength")
    ax[1].set_title("Atomic Mass against inverse root wavelength")

    print(f"Alpha 40 Fit: R = ({(4/3)*alpha_40_fit[0]**2:.5g} +/- {alpha_40_cov[0][0]}) m^(-1)\
        \n sigma_k = ({(4/3) * alpha_40_fit[1]/alpha_40_fit[0]:.3f} +/- add) \n")

    print(f"Beta 40 Fit: R = ({(9/8)*beta_40_fit[0]**2:.5g} +/- {beta_40_cov[0][0]}) m^(-1)\
        \n sigma_k = ({(9/8) * beta_40_fit[1]/beta_40_fit[0]:.3f} +/-  add) \n")

    print(f"Alpha 30 Fit: R = ({(4/3)*alpha_30_fit[0]**2:.5g} +/- {alpha_30_cov[0][0]}) m^(-1)\
        \n sigma_k = ({(4/3) * alpha_30_fit[1]/alpha_30_fit[0]:.3f} +/-  add) \n")

    print(f"Beta 30 Fit: R = ({(9/8)*beta_30_fit[0]**2:.5g} +/- {beta_30_cov[0][0]}) m^(-1)\
        \n sigma_k = ({(9/8) * beta_30_fit[1]/beta_30_fit[0]:.3f} +/-  add) \n")

    print(f"Percentage Differnce alpha 40: {100*(R_0 - (4/3)*alpha_40_fit[0]**2)/R_0}")
    print(f"Percentage Differnce beta 40: {100*(R_0 - (9/8)*beta_40_fit[0]**2)/R_0}")
    print(f"Percentage Differnce alpha 30: {100*(R_0 - (4/3)*alpha_30_fit[0]**2)/R_0}")
    print(f"Percentage Differnce beta 30: {100*(R_0 - (9/8)*beta_30_fit[0]**2)/R_0}")

    ax[0].legend(loc="lower right")
    ax[1].legend(loc="lower right")
    plt.show()


def ryd_intercept(x,y):
    c = y - np.sqrt(R_0)*x
    return c

def energy_intercept(E,Z):
    return Z - np.sqrt(E/13.6)

calculated_intercepts = np.array(ryd_intercept(Z,alpha_plot))
print(f"Screening Constants assuming Rydburg gradient {calculated_intercepts/np.sqrt(R_0)}")

fig,ax = plt.subplots(1,2,figsize=(6,6))
ax[0].scatter(Z,alpha_plot,color='black',marker='x',label=r'$K_ \alpha$')
ax[0].scatter(Z,beta_plot,color='grey',marker='x',label=r'$K_{\beta}$')
ax[0].set_xlabel("Atomic Number (Z)")
ax[0].set_ylabel(r"$\sqrt{1/\lambda}$")
ax[0].grid()
for i in range(len(calculated_intercepts)):
    ax[0].plot(Z,straight_line(Z,np.sqrt(R_0),calculated_intercepts[i]),color='purple')

ax[1].scatter(A,alpha_plot,color='black',marker='x',label='Alpha Lines')
ax[1].scatter(A,beta_plot,color='grey',marker='x',label="Beta Lines")
ax[1].set_xlabel("Atomic Weight (A)")
ax[1].set_ylabel(r"$\sqrt{1/\lambda}$")
ax[1].grid()
plt.show()


ka1 = []
kb1 = []
for i,col_name in enumerate(columns):
    for name, line in xraydb.xray_lines(elements[i], 'K').items():
        if name == 'Ka1':
            # ka1 = line.energy
            ka1.append(line.energy)
        elif name == 'Kb1':
            kb1.append(line.energy)
print(f"K_alpha energies: {ka1}")
print(f"K_beta energies: {kb1}")

alpha_intercepts = np.array(energy_intercept(Z,np.array(ka1)))
# beta_intercepts = np.array(ryd_intercept(Z,kb1))
plt.scatter(Z,calculated_intercepts/np.sqrt(R_0),color='blue')
plt.scatter(Z,-1*alpha_intercepts/np.sqrt(R_0),color='red')#/np.sqrt(R_0),color='red')
print(calculated_intercepts/np.sqrt(R_0))
print(alpha_intercepts)
plt.show()


