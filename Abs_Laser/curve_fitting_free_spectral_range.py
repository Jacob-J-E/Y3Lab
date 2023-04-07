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

def fsr_function(x,a,b,c,d):
    return (a/-(c*(x-b))) + d

lines_87_2 = np.array([3.84227691610721E+14,3.84227848551321E+14,3.84228115203221E+14])

lines_87_1 = np.array([3.84234454071332E+14,3.84234526293332E+14,3.84234683233932E+14])

lines_85_3 = np.array([3.84229057649484E+14,3.84229121049484E+14,3.84229241689484E+14])

lines_85_2 = np.array([3.84232064008923E+14,3.84232093381923E+14,3.84232156781923E+14])

cross_lines_87_2 = []
cross_lines_87_1 = []
cross_lines_85_3 = []
cross_lines_85_2 = []

for j in range(0,2):
    for k in range(j+1,3):
        cross_lines_87_2.append((lines_87_2[j] + lines_87_2[k])/2)
        cross_lines_87_1.append((lines_87_1[j] + lines_87_1[k])/2)
        cross_lines_85_3.append((lines_85_3[j] + lines_85_3[k])/2)
        cross_lines_85_2.append((lines_85_2[j] + lines_85_2[k])/2)



data = pd.read_csv(r"Abs_Laser\Data\10-03-2023\NEW1B.CSV")
data_DB_free = pd.read_csv(r"Abs_Laser\Data\10-03-2023\NEW1.CSV")

x_axis = data_DB_free['in s']
c1 = data_DB_free['C1 in V']           
c2 = data_DB_free['C2 in V'] 
c3 = data_DB_free['C3 in V']
c4 = data_DB_free['C4 in V']

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

x_axis, c3, c4,c1,c1_B = zip(*sorted(zip(x_axis, c3, c4,c1,c1_B )))

x_axis = np.array(x_axis)
c3 = np.array(c3)
c4 = np.array(c4)
c1 = np.array(c1)
c1_B = np.array(c1_B)

'''
calculating peak shifts
'''

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
    #print(para)
    plt.plot(x_axis_linspace,y_data,color='black')
    center_array.append(para[0])
    amplitude_array.append(max(y_data))
    std_array.append((4*para[1])/(2*np.sqrt(2*np.log(2))))
    cov_array.append(np.sqrt(cov[0][0]))

    if i == 1:
        print(f'1 width {para[1]}')
    elif i == 2:
        print(f'2 width {para[1]}')


print(f'FSR: {center_array[2]- center_array[1]}')

plt.scatter(center_array,amplitude_array,marker = 'x',color='green', label='Fitted Lorentzian')
plt.errorbar(center_array,amplitude_array,xerr=1*np.array(std_array),ls='None',color='green',capsize=5,label= 'Fitted Lorentzian')

FP = np.array(c4)
peak_y = np.array(amplitude_array)
peak_x = np.array(center_array)
spacing = np.diff(peak_x)

q3, q1 = np.percentile(spacing, [75 ,25])
iqr = q3 - q1
#half_peak = (peak_x[1:] + peak_x[:-1])/2
half_peak = peak_x[:-1]
peak_x_spliced = half_peak[(spacing < (q3+ 2*iqr)) & (spacing > (q1- 2*iqr)) ]
spacing_spliced = spacing[(spacing < (q3+ 2*iqr)) & (spacing > (q1- 2*iqr)) ]

inital_guess = [1.15856559e-02,1.95808248e-01 ,4.88592568e+01,0]
#inital_guess = [1.15856559e-02,1.95808248e-01 ,4.88592568e+01]
para,cov = curve_fit(fsr_function,peak_x_spliced,spacing_spliced,inital_guess)
linspace = np.linspace(min(peak_x_spliced), max(peak_x_spliced), 1000000)

print(f'para: {para}')
plt.figure()
plt.scatter(peak_x_spliced,spacing_spliced)
plt.plot(linspace,fsr_function(linspace,para[0], para[1],para[2],para[3]))

string = ''
for i in range(len(peak_x_spliced)):
    string = string + f'({peak_x_spliced[i]},{spacing_spliced[i]}),'

print(string)


fpr_scaling = fsr_function(x_axis,para[0], para[1],para[2],para[3])

scaling = []
FP_length = 2*19e-2
# FP_length = 2 * 19.5e-2
# FP_length = 2 * 20e-2


for i in range(len(fpr_scaling)):
    scaling.append(((3e8/(2*FP_length))/fpr_scaling[i]))

freq = [] 
for i in range(len(scaling)):
    freq.append(np.array(x_axis[i])*scaling[i])

freq = np.array(freq)

scaling_mean = np.mean(scaling)
freq_static = [] 
for i in range(len(scaling)):
    freq_static.append(np.array(x_axis[i])*scaling_mean)



peaks_fine, _= find_peaks(-c1_B, distance=8000)
c1_b_grouped_peaks =c1_B[peaks_fine]
freq_peaks = freq[peaks_fine]
print(f'freq_peaks {freq_peaks}')
print(f'c1_b_grouped_peaks {c1_b_grouped_peaks}')

centers_fine = [freq_peaks[1],freq_peaks[2],freq_peaks[4],freq_peaks[6]]
amplitude_fine = [c1_b_grouped_peaks[1],c1_b_grouped_peaks[2],c1_b_grouped_peaks[4],c1_b_grouped_peaks[6]]

print(f'Fine Structure Gap 1 {freq_peaks[4]-freq_peaks[2]}')
print(f'Fine Structure Gap 2 {freq_peaks[6]-freq_peaks[1]}')

plt.figure()
plt.plot(freq,c1_B, label = 'Varying scaling value (no offset)')
plt.scatter(centers_fine,amplitude_fine, marker = 'x')
plt.plot(freq_static,c1_B, label = 'Mean constant scaling Value (no offset)')
plt.legend()

plt.figure()

offset = (384.230406373e12-2.563005979089109e9) - freq_peaks[1]
offset = 3.84227921462521e14 - freq_peaks[1]
#offset = (3.84228115203221e14) - (-3.824e9)
freq_offset = freq + offset

peaks_fine, _= find_peaks(-c1_B, distance=8000)
c1_b_grouped_peaks =c1_B[peaks_fine]
freq_peaks_offset = freq_offset[peaks_fine]
print(f'freq_peaks_offset {freq_peaks_offset}')
print(f'c1_b_grouped_peaks {c1_b_grouped_peaks}')

centers_fine_offset = [freq_peaks_offset[1],freq_peaks_offset[2],freq_peaks_offset[4],freq_peaks_offset[6]]
amplitude_fine_offset = [c1_b_grouped_peaks[1],c1_b_grouped_peaks[2],c1_b_grouped_peaks[4],c1_b_grouped_peaks[6]]

print(f'Fine Structure offset Gap 1 {freq_peaks[4]-freq_peaks[2]}')
print(f'Fine Structure offset Gap 2 {freq_peaks[6]-freq_peaks[1]}')

plt.axvline(freq_peaks[4] + offset, label = '3', color = 'green')
plt.axvline(freq_peaks[2]+ offset, label = '2', color = 'green')
plt.axvline(freq_peaks[1]+ offset, label = '1', color = 'green')
plt.axvline(freq_peaks[6]+ offset, label = '4', color = 'green')

plt.axvline(3.84227921462521e14, label = '1 theory', color = 'orange')
plt.axvline(3.84229141484484e14, label = ' 2 theory', color = 'orange')
plt.axvline(3.84232177216923e14, label = ' 3 theory', color = 'orange')
plt.axvline(3.84234756145132e14, label = '4 theory', color = 'orange')


plt.plot(freq_offset,c1_B,alpha=0.5)
plt.plot(freq_offset,c1,alpha=0.5)
plt.scatter(centers_fine_offset,amplitude_fine_offset, marker = 'x')

for i in range(0,3):    
    plt.axvline(lines_85_2[i])
    plt.axvline(lines_85_3[i])
    plt.axvline(lines_87_2[i])
    plt.axvline(lines_87_1[i])
    plt.axvline(cross_lines_85_2[i],color='red')
    plt.axvline(cross_lines_85_3[i],color='red')
    plt.axvline(cross_lines_87_2[i],color='red')
    plt.axvline(cross_lines_87_1[i],color='red')

plt.figure()
plt.plot(freq_offset,c1-c1_B/max(c1_B)*max(c1)+0.2,alpha=0.5)

for i in range(0,3):    
    plt.axvline(lines_85_2[i])
    plt.axvline(lines_85_3[i])
    plt.axvline(lines_87_2[i])
    plt.axvline(lines_87_1[i])
    plt.axvline(cross_lines_85_2[i],color='red')
    plt.axvline(cross_lines_85_3[i],color='red')
    plt.axvline(cross_lines_87_2[i],color='red')
    plt.axvline(cross_lines_87_1[i],color='red')

plt.axvline(freq_peaks[4] + offset, label = '3', color = 'green')
plt.axvline(freq_peaks[2]+ offset, label = '2', color = 'green')
plt.axvline(freq_peaks[1]+ offset, label = '1', color = 'green')
plt.axvline(freq_peaks[6]+ offset, label = '4', color = 'green')

plt.axvline(3.84227921462521e14, label = '1 theory', color = 'orange')
plt.axvline(3.84229141484484e14, label = ' 2 theory', color = 'orange')
plt.axvline(3.84232177216923e14, label = ' 3 theory', color = 'orange')
plt.axvline(3.84234756145132e14, label = '4 theory', color = 'orange')



plt.show()