import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
import seaborn as sns
import scipy.optimize as spo
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
hep.style.use("ATLAS")
# from matplotlib import rc
# rc('text', usetex=True)
plt.style.use('dark_background')
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rcParams["text.usetex"]
R_0 = 10973731.6

def straight_line(x,m,c):
    return m * x + c

def cal_rydberg(m,factor):
    return (1/factor)*m**2

Z = np.arange(40,80,1)
sigma_moseley_alpha = 1
sigma_moseley_beta = 7.4
# energy = np.linspace(0,100,100) * 1e3 * 1.6e-19
# wav = (6.63e-34 * 3e8) / energy

moseley = R_0 * (Z-sigma_moseley_alpha)
K_alpha = (3/4) * R_0 * (Z-sigma_moseley_alpha)
K_beta = (8/9) * R_0 * (Z-sigma_moseley_beta)
L_alpha = (5/36)* R_0 * (Z-sigma_moseley_beta)
L_beta = (3/16)* R_0 * (Z-sigma_moseley_beta-0.2 )
L_gamma = (21/100)* R_0 * (Z-sigma_moseley_beta-0.5)

plt.figure(1,figsize=(14,10))
plt.plot(Z,moseley,label="Moseley's Law",color="#46bddf")
plt.plot(Z,K_alpha,label=r"$K_{\alpha} Line$",color="#e4e5e7")
plt.plot(Z,K_beta,label=r"$K_{\beta} Line$",color="#f05464")
plt.plot(Z,L_alpha,label=r"$L_{\alpha} Line$",color="#e87454")
plt.plot(Z,L_beta,label=r"$L_{\beta} Line$",color="#58d474")
plt.plot(Z,L_gamma,label=r"$L_{\gamma} Line$",color="#e8c454")
plt.legend(loc="upper left")
plt.xlabel("Atomic Number",fontsize=20,labelpad=10)
plt.ylabel(r"$\sqrt{\frac{1}{\lambda}}$   $\left(m^{-1/2}\right)$",fontsize=20,labelpad=10)
plt.grid(alpha=0.5)
plt.show()
