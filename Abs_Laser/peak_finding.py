import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
import scipy.interpolate as spi
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
from sklearn.preprocessing import *
from sklearn.cluster import DBSCAN
from itertools import chain
import copy
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

def lorentzian(x, center, width, amplitude, c):
    """
    Calculate the Lorentzian function for the given x values.

    Parameters
    ----------
    x : array-like
        Input x values to calculate the Lorentzian function.
    center : float
        The center of the Lorentzian function.
    width : float
        The width of the Lorentzian function (also known as FWHM: Full Width at Half Maximum).
    amplitude : float
        The amplitude of the Lorentzian function.
    c: float
        Global vertical shift.

    Returns
    -------
    y : array-like
        The Lorentzian function values corresponding to the input x values.
    """
    y = ((amplitude / np.pi) * (width / 2) / ((x - center)**2 + (width / 2)**2))+ c
    return y

data = pd.read_csv(r"Abs_Laser\Data\10-03-2023\NEW1B.CSV")
data_DB_free = pd.read_csv(r"Abs_Laser\Data\10-03-2023\NEW1.CSV")

x_axis = data_DB_free['in s']
# x_axis = np.array(x_axis)-min(x_axis)
c1 = data_DB_free['C1 in V']           
c2 = data_DB_free['C2 in V'] 
c3 = data_DB_free['C3 in V']
c4 = data_DB_free['C4 in V']

c1_B = data['C1 in V']

c1_B = c1_B/max(c1_B)*max(c1)
x2 = data['in s']

c1_B = np.array(c1_B)
c1 = np.array(c1)
x_axis = np.array(x_axis)

c1_B = c1_B[(x_axis > -0.01749) & (x_axis < 0.01573)]
c1 = c1[(x_axis > -0.01749) & (x_axis < 0.01573)]
c4 = c4[(x_axis > -0.01749) & (x_axis < 0.01573)]
c3 = c3[(x_axis > -0.01749) & (x_axis < 0.01573)]
x_axis = x_axis[(x_axis > -0.01749) & (x_axis < 0.01573)]
x_axis = x_axis[::-1]

x_axis, c3, c4,c1,c1_B = zip(*sorted(zip(x_axis, c3, c4,c1,c1_B )))
x_axis = np.array(x_axis)
c3 = np.array(c3)
c4 = np.array(c4)
c1 = np.array(c1)
c1_B = np.array(c1_B)

peaks, _= find_peaks(c4, distance=2000)
c4_peaks = c4[peaks]
x_axis_peaks = x_axis[peaks]

x_axis_linspace = np.linspace(x_axis[0], x_axis[-1], 1000000)

center_array = []
amplitude_array = []
std_array = []
cov_array = []
plt.figure()
plt.plot(x_axis,c4, label = 'Raw Data', color = 'blue')
plt.scatter(x_axis_peaks,c4_peaks, marker = 'x', label='Peak Finder', color = 'red')

for i in range(len(x_axis_peaks)):
    inital_guess = [x_axis_peaks[i], 0.00004,c4_peaks[i], 0]  

    para, cov = curve_fit(lorentzian,x_axis, c4, inital_guess)
    y_data = lorentzian(x_axis_linspace,para[0], para[1], para[2], para[3])
    print(para)
    plt.plot(x_axis_linspace,y_data,color='black')
    center_array.append(para[0])
    amplitude_array.append(max(y_data))
    std_array.append((4*para[1])/(2*np.sqrt(2*np.log(2))))
    cov_array.append(np.sqrt(cov[0][0]))

plt.scatter(center_array,amplitude_array,marker = 'x',color='green', label='Fitted Lorentzian')
plt.errorbar(center_array,amplitude_array,xerr=1*np.array(std_array),ls='None',color='green',capsize=5,label= 'Fitted Lorentzian')
plt.errorbar(center_array,amplitude_array,xerr=1*np.array(cov_array),ls='None',color='black',capsize=5,label= 'Fitted Cov')

    
plt.legend(loc='upper right')



plt.show()









