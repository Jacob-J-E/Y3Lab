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
import statsmodels.api as sm
import scipy.stats as stats


# Min Max Values are inclusive
iterations = 3
max_poly_order = 4
min_poly_order = 2
if min_poly_order > max_poly_order:
    print('Make sure max poly order is higher or equal to min poly order')
    exit()
elif min_poly_order < 2:
    print('Please put a coeff greater than or equal to 2')
    exit()


R_1_R_2 = 0.995*0.995
F = 0.5 * (4 * (R_1_R_2)**(1/4))/(1-(R_1_R_2)**(1/2))
# F = (np.pi/2)*((np.arcsin((1-np.sqrt(R_1_R_2))/(2*(R_1_R_2))**(1/4)))**(-1))
fsr_theoretical = (3e8/(4*20e-2))


print('********CONSTANTS***********')
print(f'Finesse, F: {F}')
print(f'Theoretical FSR, F: {fsr_theoretical}')
print('****************************')

def chi_squared(o,e):
    o = np.array(o)
    e = np.array(e)
    if len(o) != len(e):
        print('MAKE SURE OBSERVED AND EXPECTED ARRAY ARE EQUAL SIZES')
        exit()

    chi_squared_val = 0
    for i in range(len(o)):
        chi_squared_val += (o[i] - e[i])**2/e[i]
    
    return chi_squared_val

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

def lorentzian_normalised(x, center, width, amplitude, c):
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
    y = ((width / 2) / ((x - center)**2 + (width / 2)**2))+ c
    y = (y/max(y))*amplitude
    return y


def f(x,*coeff):
    coeff = list(coeff)
    x = np.array(x)
    function = np.zeros(len(x))
    for i in range(len(coeff)):
        function += coeff[i]*x**(i)
    return function

def airy_modified_function(x,*args):
    args = list(args)
    num = args[-2] + x*args[-1]
    f_values = f(x,*args[:-2])
    dom = 1 + F*(np.sin((np.pi/fsr_theoretical)*f_values))**2
    return (num/dom) 


def straight_line(x,a,b):
    return a*x+b

def five_lor_x(x,f1,w1,a1,f2,w2,a2,f3,w3,a3,f4,w4,a4,f5,w5,a5, a,b,c):
    return lorentzian(x,f1,w1,a1,0) + lorentzian(x,f2,w2,a2,0) + lorentzian(x,f3,w3,a3,0) + lorentzian(x,f4,w4,a4,0) + lorentzian(x,f5,w5,a5,0) - (a*x**2 + b*x) + c

def five_lor_x_update(x,f1,w1,a1,f2,w2,a2,f3,w3,a3,f4,w4,a4,f5,w5,a5,f6,w6,a6,m,c):
    return lorentzian(x,f1,w1,a1,0) + lorentzian(x,f2,w2,a2,0) + lorentzian(x,f3,w3,a3,0) + lorentzian(x,f4,w4,a4,0) + lorentzian(x,f5,w5,a5,0) +  lorentzian(x,f6,w6,a6,0) + straight_line(x,m,c)

def four_lor_x_update(x,f1,w1,a1,f2,w2,a2,f3,w3,a3,f4,w4,a4,m,c):
    return lorentzian_normalised(x,f1,w1,a1,0) + lorentzian_normalised(x,f2,w2,a2,0) + lorentzian_normalised(x,f3,w3,a3,0) + lorentzian_normalised(x,f4,w4,a4,0) + straight_line(x,m,c)

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
c3 = np.array(c3)
c4 = np.array(c4)
c1 = np.array(c1)
x_axis = np.array(x_axis)

c1_B = c1_B[(x_axis > -0.01749) & (x_axis < 0.01573)]
c1 = c1[(x_axis > -0.01749) & (x_axis < 0.01573)]
c4 = c4[(x_axis > -0.01749) & (x_axis < 0.01573)]
c3 = c3[(x_axis > -0.01749) & (x_axis < 0.01573)]
x_axis = x_axis[(x_axis > -0.01749) & (x_axis < 0.01573)]

x_axis = x_axis[::-1]
x_axis, c3, c4,c1,c1_B = zip(*sorted(zip(x_axis, c3, c4,c1,c1_B )))

c1_B = np.array(c1_B)
c3 = np.array(c3)
c4 = np.array(c4)
c1 = np.array(c1)
x_axis = np.array(x_axis)

#NORMALIZING X AXIS -------------------------------------------------------------------------------
length_of_xaxis = len(x_axis)
normalized_x_axis = []
for i in range(len(x_axis)):
    normalized_x_axis.append((i-(length_of_xaxis/2))/(length_of_xaxis/2))

normalized_x_axis = np.array(normalized_x_axis)

#NORMALIZING Y AXIS FP-------------------------------------------------------------------------------
c4 = c4 - min(c4)
c4 = c4/max(c4)
#FITTING LORENZIANS TO FP AND FINDING PEAK VALUES----------------------------------------------------
peaks, _= find_peaks(c4, distance=2000)
c4_peaks = c4[peaks]
x_axis_peaks = normalized_x_axis[peaks]
x_axis_peaks = x_axis_peaks[1:]
c4_peaks = c4_peaks[1:]
spacing = np.diff(x_axis_peaks)

lorentzian_array = np.zeros(len(normalized_x_axis))
FP_lorenzian_x_axis_peaks = []
FP_lorenzian_x_axis_peaks_amplitude = []
for i in range(len(x_axis_peaks)):
    inital_guess = [x_axis_peaks[i], 0.00004,c4_peaks[i], 0]  
    para, cov = curve_fit(lorentzian,normalized_x_axis, c4, inital_guess)
    y_data = lorentzian(normalized_x_axis,para[0], para[1], para[2], para[3])
    FP_lorenzian_x_axis_peaks.append(para[0])
    FP_lorenzian_x_axis_peaks_amplitude.append(max(y_data) - para[3])
    lorentzian_array += y_data - para[3]

FP_lorenzian_x_axis_peaks = np.array(FP_lorenzian_x_axis_peaks)
FP_lorenzian_x_axis_peaks_amplitude = np.array(FP_lorenzian_x_axis_peaks_amplitude)


#FINDING A_0 AND A_1 ----------------------------------------------------------------------------

relative_FSP_SHIFT = 0
relative_FSP_SHIFT_ARRAY = []
for i in range(len(FP_lorenzian_x_axis_peaks)):
    relative_FSP_SHIFT_ARRAY.append(relative_FSP_SHIFT)
    relative_FSP_SHIFT += fsr_theoretical

relative_FSP_SHIFT_ARRAY = np.array(relative_FSP_SHIFT_ARRAY)

m = (relative_FSP_SHIFT_ARRAY[-1] - relative_FSP_SHIFT_ARRAY[0])/ (FP_lorenzian_x_axis_peaks[-1] - FP_lorenzian_x_axis_peaks[0])
c = relative_FSP_SHIFT_ARRAY[0] - m*FP_lorenzian_x_axis_peaks[0]
inital_guess_straight = [m,c]
para_straight, cov_straight = curve_fit(straight_line,FP_lorenzian_x_axis_peaks,relative_FSP_SHIFT_ARRAY,inital_guess_straight)

print('***GUESS FOR A_1 AND A_0****')
print(f'Fitted Gradient (A_1): {para_straight[0]:.4g} +/- {cov_straight[0][0]**(0.5):.4g}')
print(f'Fitted Intercept (A_0): {para_straight[1]:.4g} +/- {cov_straight[1][1]**(0.5):.4g}')
print('****************************')

a_0_guess = para_straight[1]
a_1_guess = para_straight[0]

#Plotting modified airy function ----------------------------------------------------------------


array_of_coeffients = []
b_values = []
for i in range(min_poly_order,max_poly_order+1):
    poly_order = i 
    # ordering -> a_0,a_1, a_2, a_3, a_4, b_0, b_1)
    inital_guess_airy_modified = [a_0_guess,a_1_guess]

    for i in range(poly_order+1):
        inital_guess_airy_modified.append(0)

    with alive_bar(iterations) as bar:
        for i in range(iterations):
            bar()
            #bounds= ((0,0,0,0,0,0,0),(np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf))
            para_airy_modified, cov_airy_modified = curve_fit(airy_modified_function,normalized_x_axis,lorentzian_array,inital_guess_airy_modified)
            inital_guess_airy_modified = para_airy_modified

    a_coeffients = para_airy_modified[:poly_order+1]
    b_coeff = [para_airy_modified[-2],para_airy_modified[-1]]
    array_of_coeffients.append(a_coeffients)
    b_values.append(b_coeff)
    print('****************************')
    print(f'COMPLETED BOOTSTRAPING with Poly Order {poly_order}')
    print(f'Poly Order {poly_order} - a coeffients {a_coeffients}')
    print(f'Poly Order {poly_order} - b coeffients {b_coeff}')
    print('****************************')

from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
domain_airy_modified = np.linspace(min(normalized_x_axis),max(normalized_x_axis), 100000)
fig, ax1 = plt.subplots(figsize=(15, 15))
# plt.title('Fitted FP using Modified Airy Function')
ax2 = plt.axes([0,0,.1,.2])
ax3 = plt.axes([0,0,.1,.2])
# Manually set the position and relative size of the inset axes within ax1
size_x = .45
size_y = 0.9
ip2 = InsetPosition(ax1, [0.0,1.2,size_x,size_y])
ip3 = InsetPosition(ax1, [1-size_x,1.2,size_x,size_y])

ax2.set_axes_locator(ip2)
ax3.set_axes_locator(ip3)

mark_inset(ax1, ax2, loc1=1, loc2=3, fc="none", ec='0.5')
mark_inset(ax1, ax3, loc1=1, loc2=3, fc="none", ec='0.5')
ax2.set_title(r"$\bf{(C)}$",loc='right', y=1.05)
ax3.set_title(r"$\bf{(B)}$",loc='right', y=1.05)
title1 = ax1.set_title(r"$\bf{(A)}$",loc='right', y=0.95)

offset = np.array([+0.05, 0])
title1.set_position(title1.get_position() + offset)

c4 = c4 -0.2
down_2 = -0.1
up_2 =  -0.082
down_3 = 0.978
up_3 = 1
ax1.plot(normalized_x_axis,c4, label =  "Data" )
ax2.plot(normalized_x_axis[(normalized_x_axis < up_2) & (normalized_x_axis > down_2)], c4[(normalized_x_axis < up_2) & (normalized_x_axis > down_2)], alpha=1, label="Data")
ax3.plot(normalized_x_axis[(normalized_x_axis < up_3) & (normalized_x_axis > down_3)], c4[(normalized_x_axis < up_3) & (normalized_x_axis > down_3)], alpha=1, label="Data")

for i in range(len(array_of_coeffients)):
    para = list(array_of_coeffients[i]) + list(b_values[i])
    y_data = airy_modified_function(domain_airy_modified,*para)

    ax1.plot(domain_airy_modified,airy_modified_function(domain_airy_modified,*para), label =  f"Fitted Airy (f(x) poly order {min_poly_order + i})" )
    ax2.plot(domain_airy_modified[(domain_airy_modified < up_2) & (domain_airy_modified > down_2)],y_data[(domain_airy_modified < up_2) & (domain_airy_modified > down_2)], label =  f"Fitted Airy (f(x) poly order {min_poly_order + i})")
    ax3.plot(domain_airy_modified[(domain_airy_modified < up_3) & (domain_airy_modified > down_3)],y_data[(domain_airy_modified < up_3) & (domain_airy_modified > down_3)], label =  f"Fitted Airy (f(x) poly order {min_poly_order + i})" )


# Create a set of inset Axes: these should fill the bounding box allocated to
# them.

ax1.set_xlabel("Normalized Time Scale (Arb.)")
ax2.set_xlabel("Normalized Time Scale (Arb.)")
ax3.set_xlabel("Normalized Time Scale (Arb.)")

ax1.set_ylabel("Intensity (Arb.)")
ax2.set_ylabel("Intensity (Arb.)")
ax3.set_ylabel("Intensity (Arb.)")


ax1.legend(loc='lower left').set_zorder(100)
# ax2.legend(loc=0)
# ax3.legend(loc=0)
plt.subplots_adjust(top=.5)
plt.show()
