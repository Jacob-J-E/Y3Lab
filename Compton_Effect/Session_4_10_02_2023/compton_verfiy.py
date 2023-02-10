import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
from scipy.signal import savgol_filter
import scipy.optimize as spo
import scipy as sp 
hep.style.use("CMS")


def chi_square(obs,exp):
    obs = np.array(obs)
    exp = np.array(exp)
    chi_val = (obs-exp)**2/exp**2
    return sum(chi_val)

def energy_compton(x,E_0, m_e = 9.11e-31, c=3e8):
    return E_0 / (1+((E_0)/(511))*(1-x))


def energy_convert(x):
    return 0.36788261 * x - 10.41037249

def gaussian(x, a, b, c, e):
    return (a * np.exp(-((x - b) ** 2) / (2 * c ** 2)) + e)

data_cs = pd.read_csv(r"Compton_Effect\Data\Session_4_10_02_2023\80_degrees.csv",skiprows=0)
channel = np.array(data_cs['n_1'])
channel = energy_convert(channel)

compton = []
straight = [] 
for i in range(2,10):
    path = "Compton_Effect\Data\Session_4_10_02_2023/"+str(10*i)+"_degrees.csv"
    data = pd.read_csv(path,skiprows=0)
    compton.append(data['compton'])
    straight.append(data['straight'])

fit_means = []
for i in range(0,len(compton)):
    compton_savgol = savgol_filter(compton[i]-straight[i],window_length=301,polyorder=3)
    compton_reduced = compton[i]-straight[i]
    index = np.argmax(compton_savgol)
    params, cov = spo.curve_fit(gaussian,channel,compton_reduced,[compton_savgol[index],channel[index],10*np.std(compton_savgol),0])
    fit_means.append(params[1])

angle_plot = np.arange(20,90,0.1)
angle = np.array([20, 30 ,40 ,50 ,60 ,70 ,80, 90]) * np.pi / 180
params, cov = spo.curve_fit(energy_compton,np.cos(angle),fit_means,[622])
print(params)
chi = chi_square(fit_means,energy_compton((np.cos(angle)),E_0=params))
print("Chi Squared Value" , chi)

plt.plot(np.cos(angle),energy_compton(np.cos(angle),622))

plt.scatter(np.cos(angle),fit_means,marker='x',color='red')
plt.plot(np.cos(angle_plot),energy_compton((np.cos(angle_plot)),E_0=params))
plt.xlim(-0.2,1)
plt.xlabel(r"$\cos \theta$")
plt.ylabel("Energy (keV)")
plt.grid(alpha=0.8)
plt.show()
