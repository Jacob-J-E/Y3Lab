import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import pandas as pd
import math
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
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



def gaussian(x, a, b, c,e):
    return (a * np.exp(-((x - b) ** 2) / (2 * c ** 2)) + e)


bounds_para = [(0, np.inf),(0, np.inf),(0, np.inf),(0, np.inf)]

guess_Cs_1 = [530,1800,1,0]
params_Cs_1, cov_Cs_1 = spo.curve_fit(gaussian,bins[1530:2047],Cs[1530:2047],guess_Cs_1, bounds = (0, np.inf))

guess_Cs_2 = [300,110,1,0]
params_Cs_2, cov_Cs_2 = spo.curve_fit(gaussian,bins[:170],Cs[:170],guess_Cs_2, bounds = (0, np.inf))

guess_Cs_3 = [165,240,1,0]
params_Cs_3, cov_Cs_3 = spo.curve_fit(gaussian,bins[169:380],Cs[169:380],guess_Cs_3, bounds = (0, np.inf))

print(params_Cs_1)
print(params_Cs_2)
print(params_Cs_3)
plt.plot(bins,Cs,zorder=-1)
plt.axvline(params_Cs_1[1], color = 'black')
plt.text(params_Cs_1[1]-1, params_Cs_1[0]+params_Cs_1[3], f'({params_Cs_1[1]:.4g},{(params_Cs_1[0]+params_Cs_1[3]):.4g})', horizontalalignment='right', size='large', color='black', weight='bold')
plt.axvline(params_Cs_2[1], color = 'black')
plt.text(params_Cs_2[1]-1, params_Cs_2[0]+params_Cs_2[3], f'({params_Cs_2[1]:.4g},{(params_Cs_2[0]+params_Cs_2[3]):.4g})', horizontalalignment='left', size='large', color='black', weight='bold')
plt.axvline(params_Cs_3[1], color = 'black')
plt.text(params_Cs_3[1]-1, params_Cs_3[0]+params_Cs_3[3], f'({params_Cs_3[1]:.4g},{(params_Cs_3[0]+params_Cs_3[3]):.4g})', horizontalalignment='left', size='large', color='black', weight='bold')
plt.plot(bins,gaussian(bins,*params_Cs_1), alpha = 0.4)
plt.plot(bins,gaussian(bins,*params_Cs_2), alpha = 0.4)
plt.plot(bins,gaussian(bins,*params_Cs_3), alpha = 0.4)

plt.show()

