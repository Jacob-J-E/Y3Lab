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
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from scipy.signal import butter, filtfilt
from skimage.restoration import richardson_lucy
from scipy.signal import convolve, gaussian, fftconvolve, wiener
from scipy.optimize import minimize
from scipy.ndimage import convolve1d
from alive_progress import alive_bar

data = pd.read_csv(r"Abs_Laser\Data\10-03-2023\NEW1B.CSV")
data_DB_free = pd.read_csv(r"Abs_Laser\Data\10-03-2023\NEW1.CSV")

x_axis = data_DB_free['in s']
c1 = data_DB_free['C1 in V']           
c2 = data_DB_free['C2 in V'] 
c3 = data_DB_free['C3 in V']
c4 = data_DB_free['C4 in V']


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




c1_B = data['C1 in V']

c1_B = c1_B/max(c1_B)*max(c1)

c1_B = np.array(c1_B)
c1 = np.array(c1)
x_axis = np.array(x_axis)

c1_B = c1_B[(x_axis > -0.01749) & (x_axis < 0.01573)]
c1 = c1[(x_axis > -0.01749) & (x_axis < 0.01573)]
c4 = c4[(x_axis > -0.01749) & (x_axis < 0.01573)]
c3 = c3[(x_axis > -0.01749) & (x_axis < 0.01573)]
x_axis = x_axis[(x_axis > -0.01749) & (x_axis < 0.01573)]

x_axis = np.array(x_axis)
x_axis = x_axis[::-1]

length_of_xaxis = len(x_axis)
normalized_x_axis = []
for i in range(len(x_axis)):
    normalized_x_axis.append((i-(length_of_xaxis/2))/(length_of_xaxis/2))

normalized_x_axis = np.array(normalized_x_axis)


def x_axis_scaling(axis):
    length_of_xaxis = len(axis)
    normalized_x_axis = []
    for i in range(len(axis)):
        normalized_x_axis.append((i-(length_of_xaxis/2))/(length_of_xaxis/2))

    normalized_x_axis = np.array(normalized_x_axis)

    return normalized_x_axis


x_axis, c3, c4,c1,c1_B = zip(*sorted(zip(x_axis, c3, c4,c1,c1_B )))

x_axis = np.array(x_axis)
c3 = np.array(c3)
c4 = np.array(c4)
c1 = np.array(c1)
c1_B = np.array(c1_B)

c4 = c4 - min(c4)
c4 = c4/max(c4)


R_1_R_2 = 0.995*0.995
F = (np.pi/2)*np.arcsin((1-np.sqrt(R_1_R_2))/2*(R_1_R_2)**(1/4))
F= 2*np.pi/ -np.log(R_1_R_2)
r = np.sqrt(0.995)
F = (np.pi*np.sqrt(r))/(2*(1-r))
F = 0.5 * (np.pi * (R_1_R_2)**(1/4))/(1-(R_1_R_2)**(1/2))
print('****************************')
print(f'Finesse, F: {F}')
print('****************************')

fsr_theoretical = 3e8/(2*20e-2)

def airy_modified_function(x,a_0,a_1, a_2, a_3, a_4, b_0, b_1):
    num = b_0 + x*b_1
    f = (a_0 + a_1*x + a_2*x**2 + a_3*x**3 + a_4*x**4)
    dom = 1 + F*(np.sin((np.pi/fsr_theoretical)*f))**2
    return (num/dom) 


def straight_line(x,a,b):
    return a*x+b


peaks, _= find_peaks(c4, distance=2000)
c4_peaks = c4[peaks]
x_axis_peaks = normalized_x_axis[peaks]
x_axis_peaks = x_axis_peaks[1:]
c4_peaks = c4_peaks[1:]
spacing = np.diff(x_axis_peaks)

lorentzian_array = np.zeros(len(normalized_x_axis))

for i in range(len(x_axis_peaks)):
    inital_guess = [x_axis_peaks[i], 0.00004,c4_peaks[i], 0]  

    para, cov = curve_fit(lorentzian,normalized_x_axis, c4, inital_guess)
    y_data = lorentzian(normalized_x_axis,para[0], para[1], para[2], para[3])
    lorentzian_array += y_data - para[3]


c4 = lorentzian_array
peaks, _= find_peaks(c4, distance=2000)
c4_peaks = c4[peaks]
x_axis_peaks = normalized_x_axis[peaks]
x_axis_peaks = x_axis_peaks[1:]
c4_peaks = c4_peaks[1:]
spacing = np.diff(x_axis_peaks)

q3, q1 = np.percentile(spacing, [75 ,25])
iqr = q3 - q1
peak_x_spliced = x_axis_peaks[1:][(spacing < (q3+ 1*iqr)) & (spacing > (q1- 1*iqr)) ]
spacing_spliced = spacing[(spacing < (q3+ 1*iqr)) & (spacing > (q1- 1*iqr)) ]


m = (spacing_spliced[-1] - spacing_spliced[0])/ (peak_x_spliced[-1] - peak_x_spliced[0])
c = spacing_spliced[0] - m*peak_x_spliced[0]
inital_guess_straight = [m,c]
para_straight, cov_straight = curve_fit(straight_line,peak_x_spliced,spacing_spliced,inital_guess_straight)

print('****************************')
print(f'Fitted Gradient: {para_straight[0]:.4g} +/- {cov_straight[0][0]**(0.5):.4g}')
print('****************************')
print(f'Fitted Intercept: {para_straight[1]:.4g} +/- {cov_straight[1][1]**(0.5):.4g}')

plt.figure('A plot of the difference of spacing')
plt.scatter(peak_x_spliced,spacing_spliced)
domain_straight = np.linspace(min(peak_x_spliced),max(peak_x_spliced),100000)
plt.plot(domain_straight,straight_line(domain_straight,para_straight[0], para_straight[1]),color='orange')
plt.show()
# plt.figure(f' {i}th plot of the modifed airy function of the FP')
#inital_guess_airy_modified = [para_straight[0],para_straight[1],0,0,0,1,0,0]
#inital_guess_airy_modified = [0.925,0,0.00735,1e10,0,1,0,0]
# inital_guess_airy_modified = [0.00735,1e10,0,0,0,0.925,0]
# domain_airy_modified = np.linspace(min(normalized_x_axis),max(normalized_x_axis), 100000)

# para_total_airy = np.array([domain_airy_modified] + list(inital_guess_airy_modified))
# plt.plot(domain_airy_modified, airy_modified_function(*para_total_airy))
# plt.plot(normalized_x_axis,c4)

# para_airy_modified, cov_airy_modified = curve_fit(airy_modified_function,normalized_x_axis,c4,inital_guess_airy_modified, bounds= ((0,0,0,0,0,0,0),(np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf)))
# print(para_airy_modified)
# para_total_airy = np.array([domain_airy_modified] + list(para_airy_modified))
# plt.plot(domain_airy_modified,airy_modified_function(*para_total_airy), label =  "Fitted" )
# para_total_airy = np.array([domain_airy_modified] + list(inital_guess_airy_modified))
# plt.plot(domain_airy_modified, airy_modified_function(*para_total_airy), label =  "Inital Guess" )
# plt.plot(normalized_x_axis,c4, label =  "'Data'" )


# plt.legend(loc='upper right')

empty = ''
for i in range(len(x_axis_peaks)):
    empty = empty + f'({x_axis_peaks[i]},{c4_peaks[i]}),'

print(empty)

domain_airy_modified = np.linspace(min(normalized_x_axis),max(normalized_x_axis), 100000)
inital_guess_airy_modified = [0.00735,1e10,0,0,0,0.925,0]
for i in range(30):
    plt.figure(f' {i}th plot of the modifed airy function of the FP')
    para_airy_modified, cov_airy_modified = curve_fit(airy_modified_function,normalized_x_axis,c4,inital_guess_airy_modified, bounds= ((0,0,0,0,0,0,0),(np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf)))
    print(para_airy_modified)
    para_total_airy = np.array([domain_airy_modified] + list(para_airy_modified))
    plt.plot(domain_airy_modified,airy_modified_function(*para_total_airy), label =  "Fitted" )
    para_initial_airy = np.array([domain_airy_modified] + list(inital_guess_airy_modified))
    plt.plot(domain_airy_modified, airy_modified_function(*para_initial_airy), label =  "Inital Guess" )
    plt.plot(normalized_x_axis,c4, label =  "'Data'" )
    plt.legend(loc='upper right')
    plt.show()
    inital_guess_airy_modified = para_airy_modified

#plt.plot(normalized_x_axis,c4)
#plt.scatter(x_axis_peaks,c4_peaks)

inital_guess = [] 














