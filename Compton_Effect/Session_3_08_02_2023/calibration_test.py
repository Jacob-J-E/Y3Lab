import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import pandas as pd
import math
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from scipy.odr import ODR, Data, Model
import xraydb
import mplhep as hep
from matplotlib import pyplot as plt
import numpy as np

hep.style.use("ATLAS")

data = pd.read_csv(r"Compton_Effect\Data\Session_3_08_02_2023\Cs Co Am Ba Na Na_2.csv",skiprows=0)

bins = data['n_1']
Cs = data['N_1']
Co = data['N_2']
Am = data['N_3']
Ba = data['N_4']
Na = data['N_5']
Na_2 = data['N_6']
Na = Na.to_numpy() + Na_2.to_numpy()



def gaussian(x, a, b, c,e):
    return (a * np.exp(-((x - b) ** 2) / (2 * c ** 2)) + e)

def straight_line(x, m,c):
    return (m*x+c)

def quad_line(x, a,b,c):
    return (a*x**2 + b*x + c)


def chi_square(obs,exp):
    obs = np.array(obs)
    exp = np.array(exp)
    chi_val = (obs-exp)**2/exp**2
    return sum(chi_val)


bounds_para = [(0, np.inf),(0, np.inf),(0, np.inf),(0, np.inf)]

guess_Cs_1 = [530,1800,1,0]
params_Cs_1, cov_Cs_1 = spo.curve_fit(gaussian,bins[1530:2047],Cs[1530:2047],guess_Cs_1, bounds = (0, np.inf))

guess_Cs_2 = [300,110,1,0]
params_Cs_2, cov_Cs_2 = spo.curve_fit(gaussian,bins[:170],Cs[:170],guess_Cs_2, bounds = (0, np.inf))

guess_Cs_3 = [165,240,1,0]
params_Cs_3, cov_Cs_3 = spo.curve_fit(gaussian,bins[169:380],Cs[169:380],guess_Cs_3, bounds = (0, np.inf))


plt.figure("Cs")
plt.plot(bins,Cs,zorder=-1)
mu_cs = 2
plt.axvline(params_Cs_1[1], color = 'black')
plt.text(params_Cs_1[1]-mu_cs, params_Cs_1[0]+params_Cs_1[3], f'{params_Cs_1[1]:.4g}', horizontalalignment='right', size='large', color='black', weight='bold')
plt.axvline(params_Cs_2[1], color = 'black')
plt.text(params_Cs_2[1]+mu_cs, params_Cs_2[0]+params_Cs_2[3], f'{params_Cs_2[1]:.4g}', horizontalalignment='left', size='large', color='black', weight='bold')
plt.axvline(params_Cs_3[1], color = 'black')
plt.text(params_Cs_3[1]+mu_cs, params_Cs_3[0]+params_Cs_3[3], f'{params_Cs_3[1]:.4g}', horizontalalignment='left', size='large', color='black', weight='bold')
plt.plot(bins,gaussian(bins,*params_Cs_1), alpha = 0.7)
plt.plot(bins,gaussian(bins,*params_Cs_2), alpha = 0.7)
plt.plot(bins,gaussian(bins,*params_Cs_3), alpha = 0.7)
plt.ylim(bottom = 0)

print(f'{params_Cs_1[1]:.4g} +/- {np.sqrt(cov_Cs_1[1][1]):.4g}')
print(f'{params_Cs_2[1]:.4g} +/- {np.sqrt(cov_Cs_2[1][1]):.4g}')
print(f'{params_Cs_3[1]:.4g} +/- {np.sqrt(cov_Cs_3[1][1]):.4g}')




guess_Co_1 = [58,367,1,0]
params_Co_1, cov_Co_1 = spo.curve_fit(gaussian,bins,Co,guess_Co_1, bounds = (0, np.inf))
plt.figure("Co")
plt.plot(bins,Co)
plt.text(params_Co_1[1]-1, params_Co_1[0]+params_Co_1[3], f'{params_Co_1[1]:.4g}', horizontalalignment='left', size='large', color='black', weight='bold')
plt.axvline(params_Co_1[1], color = 'black')
plt.plot(bins,gaussian(bins,*params_Co_1), alpha = 0.7)
plt.ylim(bottom = 0)


print(f'{params_Co_1[1]:.4g} +/- {np.sqrt(cov_Co_1[1][1]):.4g}')


guess_Am_1 = [3430,190,1,0]
params_Am_1, cov_Am_1 = spo.curve_fit(gaussian,bins,Am,guess_Am_1, bounds = (0, np.inf))

guess_Am_2 = [600,100,1,0]
params_Am_2, cov_Am_2 = spo.curve_fit(gaussian,bins,Am,guess_Am_2, bounds = (0, np.inf))
plt.figure("Am")
plt.plot(bins,Am)
plt.text(params_Am_1[1]-1, params_Am_1[0]+params_Am_1[3], f'{params_Am_1[1]:.4g}', horizontalalignment='left', size='large', color='black', weight='bold')
plt.axvline(params_Am_1[1], color = 'black')
plt.plot(bins,gaussian(bins,*params_Am_1), alpha = 0.7)
plt.text(params_Am_2[1]-1, params_Am_2[0]+params_Am_2[3], f'{params_Am_2[1]:.4g}', horizontalalignment='left', size='large', color='black', weight='bold')
plt.axvline(params_Am_2[1], color = 'black')
plt.plot(bins,gaussian(bins,*params_Am_2), alpha = 0.7)
plt.ylim(bottom = 0)

print(f'{params_Am_1[1]:.4g} +/- {np.sqrt(cov_Am_1[1][1]):.4g}')
print(f'{params_Am_2[1]:.4g} +/- {np.sqrt(cov_Am_2[1][1]):.4g}')

guess_Na_1 = [180,1400,1,0]
params_Na_1, cov_Na_1 = spo.curve_fit(gaussian,bins,Na,guess_Na_1, bounds = (0, np.inf))
plt.figure("Na")
plt.plot(bins,Na)
plt.text(params_Na_1[1]-1, params_Na_1[0]+params_Na_1[3], f'{params_Na_1[1]:.4g}', horizontalalignment='left', size='large', color='black', weight='bold')
plt.axvline(params_Na_1[1], color = 'black')
plt.plot(bins,gaussian(bins,*params_Na_1), alpha = 0.7)
plt.ylim(bottom = 0)

print(f'{params_Na_1[1]:.4g} +/- {np.sqrt(cov_Na_1[1][1]):.4g}')


plt.figure(5)
plt.plot(bins,Na, label = 'Na')
plt.plot(bins,Am, label = 'Am')
plt.plot(bins,Co, label = 'Co')
plt.plot(bins,Cs, label = 'Cs')

plt.axvline(params_Cs_1[1], color = 'black')
plt.text(params_Cs_1[1]-mu_cs, params_Cs_1[0]+params_Cs_1[3], f'{params_Cs_1[1]:.4g}', horizontalalignment='right', size='large', color='black', weight='bold')
plt.axvline(params_Cs_2[1], color = 'black')
plt.text(params_Cs_2[1]+mu_cs, params_Cs_2[0]+params_Cs_2[3], f'{params_Cs_2[1]:.4g}', horizontalalignment='left', size='large', color='black', weight='bold')
plt.axvline(params_Cs_3[1], color = 'black')
plt.text(params_Cs_3[1]+mu_cs, params_Cs_3[0]+params_Cs_3[3], f'{params_Cs_3[1]:.4g}', horizontalalignment='left', size='large', color='black', weight='bold')

plt.text(params_Co_1[1]-1, params_Co_1[0]+params_Co_1[3], f'{params_Co_1[1]:.4g}', horizontalalignment='left', size='large', color='black', weight='bold')
plt.axvline(params_Co_1[1], color = 'black')

plt.text(params_Am_1[1]-1, params_Am_1[0]+params_Am_1[3], f'{params_Am_1[1]:.4g}', horizontalalignment='left', size='large', color='black', weight='bold')
plt.axvline(params_Am_1[1], color = 'black')
plt.text(params_Am_2[1]-1, params_Am_2[0]+params_Am_2[3], f'{params_Am_2[1]:.4g}', horizontalalignment='right', size='large', color='black', weight='bold')
plt.axvline(params_Am_2[1], color = 'black')

plt.text(params_Na_1[1]-1, params_Na_1[0]+params_Na_1[3], f'{params_Na_1[1]:.4g}', horizontalalignment='left', size='large', color='black', weight='bold')
plt.axvline(params_Na_1[1], color = 'black')

plt.ylim(bottom = 0)
plt.xlim(left = 0 , right = 2047)
plt.legend()


bin_numbers_energy = np.array([102.4,195.9,243,369.7,1422,1822])
bin_numbers_energy_error = np.array([2.517,0.07103,0.9579,0.2318,1.56,0.3189])
energy_vals = np.array([30.85,59.54,81.0,122,511,661.7])

m = (energy_vals[-1]-energy_vals[0])/(bin_numbers_energy[-1]-bin_numbers_energy[0])
c = energy_vals[-1] - m*bin_numbers_energy[-1]
straight_line_guess = [m,c]
params_sl_3, cov_sl_3 = spo.curve_fit(straight_line,bin_numbers_energy,energy_vals,straight_line_guess)
print(params_sl_3)
domain = np.arange(0,max(bins),1)

print(f' LEAST SQUARES: straight line parameters: [{np.sqrt(cov_sl_3[0][0])},{np.sqrt(cov_sl_3[1][1])}]')
print(f' LEAST SQUARES: Error in straight line parameters: {params_sl_3}')


quad_line_guess = [0,params_sl_3[0],params_sl_3[1]]
params_ql_3, cov_ql_3 = spo.curve_fit(quad_line,bin_numbers_energy,energy_vals,quad_line_guess)
print(params_ql_3)




print(f' Chi Sqaured Value for the straight line fit: {chi_square(energy_vals,straight_line(bin_numbers_energy,*params_sl_3))}')
print(f' Chi Sqaured Value for the quadratic line fit: {chi_square(energy_vals,quad_line(bin_numbers_energy,*params_ql_3))}')

min_m = params_sl_3[0] - 3*np.sqrt(cov_sl_3[0][0])
max_m = params_sl_3[0] + 3*np.sqrt(cov_sl_3[0][0])
max_c = params_sl_3[1] + 3*np.sqrt(cov_sl_3[1][1])
min_c = params_sl_3[1] - 3*np.sqrt(cov_sl_3[1][1])


def straight_odr(B, x):
    return B[0]*x + B[1]

straight_line_model = Model(straight_odr)
data_for_cal = Data(bin_numbers_energy, energy_vals, wd=1./bin_numbers_energy_error)
myodr = ODR(data_for_cal, straight_line_model, beta0=straight_line_guess)
myoutput = myodr.run()

print(f' ODR: straight line parameters: {np.diag(myoutput.beta)}')
print(f' ODR: Error in straight line parameters: {np.sqrt(np.diag(myoutput.cov_beta))}')

plt.figure(6)
plt.scatter(bin_numbers_energy,energy_vals, marker = 'x', label = 'Data')
plt.plot(domain,straight_line(domain,*params_sl_3), label = 'Straight line Fit', color = 'green')
plt.fill_between(domain, min_m*domain + max_c , max_m*domain + min_c, label = 'Error Straight Line', alpha = 0.5, color = 'green')
plt.plot(domain,quad_line(domain,*params_ql_3), label = 'Quadratic line Fit', color = 'red')
plt.xlim(left = 0, right = max(bins))
plt.ylim(bottom = 0)
plt.legend()



plt.show()

