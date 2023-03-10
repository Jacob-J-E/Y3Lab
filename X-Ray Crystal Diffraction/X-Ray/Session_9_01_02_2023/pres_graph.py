import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import seaborn as sns
import scipy.optimize as spo
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
hep.style.use("ATLAS")
plt.style.use('dark_background')


data = pd.read_csv(r"X-Ray\Data\24-01-2023\Filter Mo Source  2nd Run.csv",skiprows=0)
print(data.columns)


energy = data['E_1 / keV']
mo_cal = data['Mo Cal']



# plt.figure(1,figsize=(10*1.618,10))
plt.figure(1)
plt.plot(energy[:-1],mo_cal[:-1],color='#44a4f4')
plt.xlim(left = 0)
plt.ylim(bottom = 0)
plt.ylabel(r'Counts $(s^{-1})$')
plt.xlabel(r'Energy $(keV)$')
#plt.title('Mo Straight Through')
plt.grid(alpha=0.5)
plt.savefig(r'X-Ray\Session_9_01_02_2023\pres_images\Mo Straight Through.jpg', dpi = 400, format = 'jpg')




mo_zr_filter = data['Mo Pure with Zr Filter']
# plt.figure(2,figsize=(10*1.618,10))
plt.figure(2)
plt.plot(energy[:-1],mo_zr_filter[:-1],color='#44a4f4')
plt.xlim(left = 0)
plt.ylim(bottom = 0)
plt.ylabel(r'Counts $(s^{-1})$')
plt.xlabel(r'Energy $(keV)$')
#plt.title('Mo straight through with Zr Filter')
plt.grid(alpha=0.5)
plt.savefig(r'X-Ray\Session_9_01_02_2023\pres_images\Mo straight through with Zr Filter.jpg', dpi = 400, format = 'jpg')

mo__plate_zr_filter = data['Mo plate with Zr Filter']
# plt.figure(3,figsize=(10*1.618,10))
plt.figure(3)
plt.plot(energy[:-1],mo__plate_zr_filter[:-1],color='#44a4f4')
plt.xlim(left = 0)
plt.ylim(bottom = 0)
plt.ylabel(r'Counts $(s^{-1})$')
plt.xlabel(r'Energy $(keV)$')
#plt.title('Mo plate with Zr Filter')
plt.grid(alpha=0.5)
plt.savefig(r'X-Ray\Session_9_01_02_2023\pres_images\Mo plate with Zr Filter.jpg', dpi = 400, format = 'jpg')


plt.figure(4)
plt.plot(energy[:-1],mo_cal[:-1]/sum(mo_cal[:-1]), label = 'Mo Straight Through')
plt.plot(energy[:-1],mo_zr_filter[:-1]/sum(mo_zr_filter[:-1]), label = 'Mo straight through with Zr Filter')
plt.plot(energy[:-1],mo__plate_zr_filter[:-1]/sum(mo__plate_zr_filter[:-1]), label = 'Mo plate with Zr Filter')
plt.xlim(left = 0)
plt.ylim(bottom = 0)
plt.ylabel(r'Normalised Counts')
plt.xlabel(r'Energy $(keV)$')
plt.legend()
#plt.title('three lines 1 graph')
plt.grid(alpha=0.5)
plt.savefig(r'X-Ray\Session_9_01_02_2023\pres_images\three_lines_1_graph.jpg', dpi = 400, format = 'jpg')
# angle = data['angle']
# wav = data['wav / pm']
# energy = data['E / keV']
# count_0 = data['R_0 / 1/s']