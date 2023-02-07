import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
import seaborn as sns
import scipy.optimize as spo
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
hep.style.use("ATLAS")
plt.style.use('dark_background')
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rcParams["text.usetex"]
R_0 = 10973731.6

def straight_line(x,m,c):
    return m * x + c

def cal_rydberg(m,factor):
    return (1/factor)*m**2

Z = np.arange(30,90,1)
Z_plot = Z[Z<80]
sigma_moseley_alpha = 1
sigma_moseley_beta = 7.4
# energy = np.linspace(0,100,100) * 1e3 * 1.6e-19
# wav = (6.63e-34 * 3e8) / energy

moseley = R_0 * (Z_plot-sigma_moseley_alpha)
K_alpha = (3/4) * R_0 * (Z_plot-sigma_moseley_alpha)
K_beta = (8/9) * R_0 * (Z_plot-sigma_moseley_beta)
L_alpha = (5/36)* R_0 * (Z_plot-sigma_moseley_beta)
L_beta = (3/16)* R_0 * (Z_plot-sigma_moseley_beta-0.2 )
L_gamma = (21/100)* R_0 * (Z_plot-sigma_moseley_beta-0.5)

plt.figure(1,figsize=(14,10))
plt.plot(Z_plot,moseley,label="Moseley's Law",color="#46bddf")
plt.plot(Z_plot,K_alpha,label=r"$K_{\alpha} Line$",color="#e4e5e7")
plt.plot(Z_plot,K_beta,label=r"$K_{\beta} Line$",color="#f05464")
plt.plot(Z_plot,L_alpha,label=r"$L_{\alpha} Line$",color="#e87454")
plt.plot(Z_plot,L_beta,label=r"$L_{\beta} Line$",color="#58d474")
plt.plot(Z_plot,L_gamma,label=r"$L_{\gamma} Line$",color="#e8c454")
# plt.legend(loc="upper left")
plt.xlabel("Atomic Number",fontsize=20,labelpad=10)
plt.ylabel(r"$\sqrt{\frac{1}{\lambda}}$   $\left(m^{-1/2}\right)$",fontsize=20,labelpad=10)



plt.text(Z_plot[-1]+1.2, moseley[-1], r"Moseley's Law", horizontalalignment='left', fontsize=18, color=u'#46bddf', weight='semibold')
plt.text(Z_plot[-1] + 1, K_alpha[-1], r'$K_{\alpha}$', horizontalalignment='left', fontsize=18, color=u'#e4e5e7', weight='semibold')
plt.text(Z_plot[-1]+ 1, K_beta[-1], r'$K_{\beta}$', horizontalalignment='left', fontsize=18, color=u'#f05464', weight='bold')

plt.text((Z_plot[-1] + 1),(L_alpha[-1]-0.2*1e8), r'$L_{\alpha}$', horizontalalignment='left', fontsize=18, color=u'#e87454', weight='bold')
plt.text((Z_plot[-1] + 1),(L_beta[-1]+0*1e8), r'$L_{\beta}$', horizontalalignment='left', fontsize=18, color=u'#58d474', weight='bold')
plt.text((Z_plot[-1] + 1),(L_gamma[-1]+0.3*1e8), r'$L_{\gamma}$', horizontalalignment='left', fontsize=18, color=u'#e8c454', weight='bold')



plt.xlim(25,92)
plt.ylim(0,1e9)
plt.grid(alpha=0.5)
plt.savefig(r"X-Ray\Plots and Figs\theo_moseley.png",dpi=400,format='png')
plt.show()
