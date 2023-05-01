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
from sklearn.metrics import r2_score 


# Min Max Values are inclusive
iterations = 20
#upp
max_poly_order = 4

#Do not change lower bound
min_poly_order = 2
if min_poly_order > max_poly_order:
    print('Make sure max poly order is higher or equal to min poly order')
    exit()
elif min_poly_order < 2:
    print('Please put a coeff greater than or equal to 2')
    exit()


R_1_R_2 = 0.995*0.995
F = 0.5 * (4 * (R_1_R_2)**(1/4))/(1-(R_1_R_2)**(1/2))
F = (np.pi/2)*((np.arcsin((1-np.sqrt(R_1_R_2))/(2*(R_1_R_2))**(1/4)))**(-1))
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
        chi_squared_val += (o[i] - e[i])**2/e[i]**2
    
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


plt.figure()
domain_airy_modified = np.linspace(min(normalized_x_axis),max(normalized_x_axis), 100000)

plt.title('Fitted FP using Modified Airy Function')
for i in range(len(array_of_coeffients)):
    para = list(array_of_coeffients[i]) + list(b_values[i])
    plt.plot(domain_airy_modified,airy_modified_function(domain_airy_modified,*para), label =  f"Fitted Airy (f(x) poly order {min_poly_order + i})" )
plt.plot(normalized_x_axis,lorentzian_array, label =  "Data" )
plt.legend()

plt.figure()
plt.title('Frequency Scaled FP')
for i in range(len(array_of_coeffients)):
    freqency_array = f(normalized_x_axis,*array_of_coeffients[i])
    plt.plot(freqency_array,lorentzian_array, label =  f"Frequency Scaling (poly order {min_poly_order + i})" )
plt.xticks(np.arange(0, max(freqency_array)+fsr_theoretical, fsr_theoretical))
plt.legend()

# plt.figure()
# height_title = 1.05
# heights = [1,1]
# fig = plt.figure()
# gs = fig.add_gridspec(2, max_poly_order, hspace=0, wspace=0.2,height_ratios=heights)
# ax = gs.subplots(sharey=False, sharex=True)
# fig.suptitle('Calibration Graph')

# colors = plt.cm.rainbow(np.linspace(0, 1, len(array_of_coeffients)))

# peaks, _= find_peaks(c4, distance=2000)
# c4_peaks = c4[peaks]
# x_axis_peaks = normalized_x_axis[peaks]
# x_axis_peaks = x_axis_peaks[1:]
# space = np.arange(0,len(x_axis_peaks),1)
# m = (x_axis_peaks[-1] - x_axis_peaks[0])/ (space[-1] - space[0])
# c = x_axis_peaks[0] - m*space[0]
# inital_guess_straight = [m,c]

# para,cov = curve_fit(straight_line,space,x_axis_peaks,inital_guess_straight)
# perr = np.sqrt(np.diag(cov))
# nstd = 3
# para_up = para + nstd * perr
# para_down = para - nstd * perr

# fit_up = straight_line(space, *para_up)
# fit_dw = straight_line(space, *para_down)
# ax[0,0].fill_between(space, fit_up, fit_dw, alpha=.25, label=str(nstd) + r'$\sigma$ Interval', color = 'orange')


# domain = np.linspace(min(space),max(space),100000)
# ax[0,0].scatter(space,x_axis_peaks, label = 'Original Data',color='orange')
# ax[0,0].plot(domain,straight_line(domain,*para), label = r'Best Fit $R^{2}$: '+f'{r2_score(x_axis_peaks,straight_line(space,*para)):.4g}', color='black')
# ax[0,0].set_title(r"$\bf{(A)}$" + ' Original Data',loc='left', y=height_title)
# ax[1,0].set_ylim(min(x_axis_peaks)-0.1,(max(x_axis_peaks)+0.1))
# ax[0,0].legend(loc='best')
# print(f'r squared: {r2_score(x_axis_peaks,straight_line(space,*para))}')

# residuals_ = x_axis_peaks - straight_line(space,*para)
# residuals_upper_ = fit_up - straight_line(space,*para)
# residuals_lower_ = fit_dw - straight_line(space,*para)

# for i in range(len(residuals_)):
#     ax[1,0].vlines(space[i], 0, residuals_[i], color = 'black')

# ax[1,0].scatter(space, residuals_, label = 'Residuals',color='orange',zorder = 2)
# ax[1,0].fill_between(space, residuals_lower_, residuals_upper_, alpha=.1, label=str(nstd) + r'$\sigma$ interval', color = 'orange')
# ax[1,0].set_ylim(min(residuals_lower_)-0.02,(max(residuals_upper_)+0.02))
# ax[1,0].hlines(0, min(space), max(space), linestyle="dashed", color = 'black', alpha = 0.5)
# ax[1,0].set_xlabel('Peak Number (No Units)')
# ax[1,0].set_ylabel('Timescale (s)')
# ax[0,0].set_ylabel('Timescale (s)')

# ax[1,0].legend(loc='best')
# # ax[1,0].grid()
# # ax[0,0].grid()
# letters = [chr(i) for i in range(ord('B'), ord('Z')+1)]
# for i in range(len(array_of_coeffients)):
#     j = i +1
#     # ax[0,j].grid()
#     # ax[1,j].grid()
#     ax[0,j].set_title(r"$\bf{(" + str(letters[i]) + ")}$"+ f' Polyorder {i+min_poly_order}',loc='left', y=height_title)
#     freqency_array = f(normalized_x_axis,*array_of_coeffients[i])
#     freq_peaks = freqency_array[peaks]
#     freq_peaks = freq_peaks[1:]


#     m = (freq_peaks[-1] - freq_peaks[0])/ (space[-1] - space[0])
#     c = freq_peaks[0] - m*space[0]
#     inital_guess_straight = [m,c]

#     para,cov = curve_fit(straight_line,space,freq_peaks,inital_guess_straight)
#     perr = np.sqrt(np.diag(cov))
#     nstd = 3
#     para_up = para + nstd * perr
#     para_down = para - nstd * perr

#     fit_up = straight_line(space, *para_up)
#     fit_dw = straight_line(space, *para_down)
#     ax[0,j].fill_between(space, fit_up, fit_dw, alpha=.25, label= str(nstd) + r'$\sigma$ Residual Interval', color = colors[i])

    
#     # plt.scatter(x_axis_peaks,freq_peaks, label = 'Data')
#     colors = ['b','g','r', 'purple']
#     ax[0,j].scatter(space,freq_peaks, label = 'Data Polyorder '+str(i+min_poly_order),color=colors[i])
#     ax[0,j].plot(domain,straight_line(domain,*para), label = r'Best Fit $R^{2}$: '+f'{r2_score(freq_peaks,straight_line(space,*para)):.4g}',color='black')
#     ax[0,j].legend(loc='best')
#     ax[0,j].set_ylim(min(freq_peaks)-1e9,max(freq_peaks)+1e9)
#     print(f'r squared: {r2_score(freq_peaks,straight_line(space,*para))}')

#     residuals_ = freq_peaks - straight_line(space,*para)
#     residuals_upper_ = fit_up - straight_line(space,*para)
#     residuals_lower_ = fit_dw - straight_line(space,*para)
#     for k in range(len(residuals_)):
#         ax[1,j].vlines(space[k], 0, residuals_[k], color = 'black')

#     ax[1,j].scatter(space, residuals_, label = 'Residuals',color=colors[i],zorder = 2)
#     ax[1,j].fill_between(space, residuals_lower_, residuals_upper_, alpha=.1, label=str(nstd) + r'$\sigma$ Residual Interval', color=colors[i])

#     ax[1,j].set_ylim(min(residuals_lower_)-1e6,max(residuals_upper_)+1e6)

#     ax[1,j].hlines(0, min(space), max(space), linestyle="dashed", color = 'black', alpha = 0.5)


#     ax[1,j].set_xlabel('Peak Number (No Units)')
#     ax[1,j].set_ylabel('Frequency (Hz)')
#     ax[0,j].set_ylabel('Frequency (Hz)')
#     ax[1,j].legend(loc='best')
    
plt.figure()
height_title = 1.05
# fig = plt.figure()
# gs = fig.add_gridspec(1, max_poly_order, wspace=0.2,)
# ax = gs.subplots(sharey=False, sharex=True)
fig,ax = plt.subplots(1,max_poly_order)

colors = plt.cm.rainbow(np.linspace(0, 1, len(array_of_coeffients)))

peaks, _= find_peaks(c4, distance=2000)
c4_peaks = c4[peaks]
x_axis_peaks = normalized_x_axis[peaks]
x_axis_peaks = x_axis_peaks[1:]
space = np.arange(0,len(x_axis_peaks),1)
m = (x_axis_peaks[-1] - x_axis_peaks[0])/ (space[-1] - space[0])
c = x_axis_peaks[0] - m*space[0]
inital_guess_straight = [m,c]

para,cov = curve_fit(straight_line,space,x_axis_peaks,inital_guess_straight)
perr = np.sqrt(np.diag(cov))
nstd = 3
para_up = para + nstd * perr
para_down = para - nstd * perr

fit_up = straight_line(space, *para_up)
fit_dw = straight_line(space, *para_down)


domain = np.linspace(min(space),max(space),100000)
ax[0].set_title(r"$\bf{(A)}$" + ' Original Data',loc='right', y=height_title)
ax[0].set_ylim(min(x_axis_peaks)-0.1,(max(x_axis_peaks)+0.1))
print(f'r squared: {r2_score(x_axis_peaks,straight_line(space,*para))}')

residuals_ = x_axis_peaks - straight_line(space,*para)
residuals_upper_ = fit_up - straight_line(space,*para)
residuals_lower_ = fit_dw - straight_line(space,*para)

for i in range(len(residuals_)):
    ax[0].vlines(space[i], 0, residuals_[i], color = 'black')

ax[0].scatter(space, residuals_, label = 'Residuals',color='orange',zorder = 2)
ax[0].fill_between(space, residuals_lower_, residuals_upper_, alpha=.1, label=str(nstd) + r'$\sigma$ interval', color = 'orange')
ax[0].set_ylim(min(residuals_lower_)-0.02,(max(residuals_upper_)+0.02))
ax[0].hlines(0, min(space), max(space), linestyle="dashed", color = 'black', alpha = 0.5)
ax[0].set_xlabel('Peak Number (No Units)')
ax[0].set_ylabel('Timescale (s)')

ax[0].legend(loc='best')
# ax[1,0].grid()
# ax[0,0].grid()
letters = [chr(i) for i in range(ord('B'), ord('Z')+1)]
for i in range(len(array_of_coeffients)):
    j = i +1
    # ax[0,j].grid()
    # ax[1,j].grid()
    ax[j].set_title(r"$\bf{(" + str(letters[i]) + ")}$"+ f' Polyorder {i+min_poly_order}',loc='right', y=height_title)
    freqency_array = f(normalized_x_axis,*array_of_coeffients[i])
    freq_peaks = freqency_array[peaks]
    freq_peaks = freq_peaks[1:]


    m = (freq_peaks[-1] - freq_peaks[0])/ (space[-1] - space[0])
    c = freq_peaks[0] - m*space[0]
    inital_guess_straight = [m,c]

    para,cov = curve_fit(straight_line,space,freq_peaks,inital_guess_straight)
    perr = np.sqrt(np.diag(cov))
    nstd = 3
    para_up = para + nstd * perr
    para_down = para - nstd * perr

    fit_up = straight_line(space, *para_up)
    fit_dw = straight_line(space, *para_down)

   
    # plt.scatter(x_axis_peaks,freq_peaks, label = 'Data')
    colors = ['b','g','r', 'purple']
    print(f'r squared: {r2_score(freq_peaks,straight_line(space,*para))}')

    residuals_ = freq_peaks - straight_line(space,*para)
    residuals_upper_ = fit_up - straight_line(space,*para)
    residuals_lower_ = fit_dw - straight_line(space,*para)
    for k in range(len(residuals_)):
        ax[j].vlines(space[k], 0, residuals_[k], color = 'black')

    ax[j].scatter(space, residuals_, label = 'Residuals',color=colors[i],zorder = 2)
    ax[j].fill_between(space, residuals_lower_, residuals_upper_, alpha=.1, label=str(nstd) + r'$\sigma$ Residual Interval', color=colors[i])

    ax[j].set_ylim(min(residuals_lower_)-1e6,max(residuals_upper_)+1e6)

    ax[j].hlines(0, min(space), max(space), linestyle="dashed", color = 'black', alpha = 0.5)


    ax[j].set_xlabel('Peak Number (No Units)')
    ax[j].set_ylabel('Relative Frequency (Hz)')
    ax[j].legend(loc='best')



plt.tight_layout()
plt.show()


