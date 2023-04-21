import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
import scipy.optimize as spo
import scipy.interpolate as spi
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
from sklearn.preprocessing import *
hep.style.use("CMS")

def line(x,m,c):
    return m*x + c

voltage = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

B_field = np.array([7.51, 8.33, 9.15, 9.97, 10.80, 11.61, 12.43, 13.26, 14.07, 14.89, 15.70, 16.54, 17.36, 18.17, 19.00])
B_field = B_field - 7

initial_guess = [(B_field[7]-B_field[2])/(voltage[7]-voltage[2]),0]
params, cov = spo.curve_fit(line,voltage,B_field,initial_guess)
print(params)
plt.figure(figsize=(10,10))
plt.scatter(voltage,B_field,label="Experimental Data",marker='x',zorder=10)
plt.plot(voltage,line(voltage, *params),label="Straight Line Fit",color='orange',zorder=0)
plt.errorbar(voltage,B_field,yerr=np.zeros_like(B_field)+0.05,capsize=7,ls='None',color='black')
plt.xlabel("Voltage (V)")
plt.ylabel("Magnetic Field (mT)")
plt.legend(loc="upper left")
plt.show()




