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


plt.figure()
domain_airy_modified = np.linspace(min(normalized_x_axis),max(normalized_x_axis), 100000)

plt.title('Fitted FP using Modified Airy Function')
for i in range(len(array_of_coeffients)):
    para = list(array_of_coeffients[i]) + list(b_values[i])
    plt.plot(domain_airy_modified,airy_modified_function(domain_airy_modified,*para), label =  f"Fitted Airy (f(x) poly order {min_poly_order + i})" )
    print(f'poly order - {min_poly_order + i} r2: {r2_score(lorentzian_array,airy_modified_function(normalized_x_axis,*para))}')
plt.plot(normalized_x_axis,lorentzian_array, label =  "Data" )
plt.legend()

plt.figure()
plt.title('Frequency Scaled FP')
for i in range(len(array_of_coeffients)):
    freqency_array = f(normalized_x_axis,*array_of_coeffients[i])
    plt.plot(freqency_array,lorentzian_array, label =  f"Frequency Scaling (poly order {min_poly_order + i})" )
plt.xticks(np.arange(0, max(freqency_array)+fsr_theoretical, fsr_theoretical))
plt.legend()



plt.figure()
plt.title('Frequency Scaled FP Peak Spacings')
colors = plt.cm.rainbow(np.linspace(0, 1, len(array_of_coeffients)))
for i in range(len(array_of_coeffients)):
    freqency_array = f(normalized_x_axis,*array_of_coeffients[i])
    freq_peaks = freqency_array[peaks]
    freq_peaks = freq_peaks[1:]
    mid_val = (freq_peaks[1:] + freq_peaks[:-1])/2
    spacing_freq = np.diff(freq_peaks)

    x = sm.add_constant(mid_val)
    model = sm.OLS(spacing_freq, x).fit()
    
    # Obtain the estimated slope (beta1) and its standard error (SE_beta1)
    beta1 = model.params[1]
    SE_beta1 = model.bse[1]

    # Calculate the fitted values
    y_fitted = model.predict(x)
    plt.plot(x[:, 1], y_fitted, color=colors[i], label='Fitted line', alpha = 0.5)

    plt.scatter(mid_val,spacing_freq, label = f'Frequency FP spacing difference ({(max(spacing_freq) - min(spacing_freq)):.4g})[poly order {min_poly_order + i}]', color = colors[i])
    plt.axhline(max(spacing_freq), color = colors[i])
    plt.axhline(min(spacing_freq), color = colors[i])
plt.legend()

plt.figure()
plt.title('T-statistic vs Poly Order (Testing Linearity)')
colors = plt.cm.hsv(np.linspace(0, 1, len(array_of_coeffients)+1))
mid_point = (x_axis_peaks[1:] + x_axis_peaks[:-1]) / 2

x = sm.add_constant(mid_point)
model = sm.OLS(spacing, x).fit()

# Obtain the estimated slope (beta1) and its standard error (SE_beta1)
beta1 = model.params[1]
SE_beta1 = model.bse[1]

# Calculate the t-statistic
t = beta1 / SE_beta1
print('**********')
print(f'T VALUE FOR (Original FP Spacing): {t:.4g}')
print('**********')
#plt.axvline(t, color = colors[0], label = f'T-value: {t} (Original FP Spacing)')

df = len(spacing) - 2

alpha_5 = 0.05
alpha_1 = 0.01
alpha_values = np.linspace(0,1,10000000)

critical_t_5_upper = stats.t.ppf(q=(1-(alpha_5/2)), df= df)
critical_t_1_upper = stats.t.ppf(q=(1 - (alpha_1/2)), df= df)
critical_t_5_lower = stats.t.ppf(q=alpha_5/2, df= df)
critical_t_1_lower = stats.t.ppf(q=alpha_1/2, df= df)

x_ = np.linspace(stats.t.ppf(0.0001, df), stats.t.ppf(0.9999, df), 10000)
y_ = stats.t.pdf(x_, df)
plt.plot(x_, y_, label = 't distribution', color = 'black')

plt.axvline(critical_t_5_upper, color = 'purple',label = f'Critical T-Value at 95% confidence (Two-Tailed) T: [{critical_t_5_upper}]')
plt.axvline(critical_t_5_lower,color = 'purple',)
# plt.axvline(critical_t_1_upper, color = 'blue', label = f'Critical T-Value at 99% confidence (Two-Tailed) T: [{critical_t_1_upper}]')
# plt.axvline(critical_t_1_lower, color = 'blue')
plt.fill_between(x_,y_,0,where=(x_>=critical_t_5_lower) & (x_<=critical_t_5_upper),color='green', label = r'Do not reject $H_{0}$ at 95% confidence', alpha=0.5)
plt.fill_between(x_,y_,0,where=(x_>=min(x_)) & (x_<=critical_t_5_lower),color='red', label = r'Reject $H_{0}$ at 95% confidence', alpha=0.5)
plt.fill_between(x_,y_,0,where=(x_>=critical_t_5_upper) & (x_<=max(x_)),color='red',alpha=0.5)


for i in range(len(array_of_coeffients)):
    freqency_array = f(normalized_x_axis,*array_of_coeffients[i])
    freq_peaks = freqency_array[peaks]
    freq_peaks = freq_peaks[1:]
    mid_val = (freq_peaks[1:] + freq_peaks[:-1])/2
    spacing_freq = np.diff(freq_peaks)

    x = sm.add_constant(mid_val)
    model = sm.OLS(spacing_freq, x).fit()
    
    # Obtain the estimated slope (beta1) and its standard error (SE_beta1)
    beta1 = model.params[1]
    SE_beta1 = model.bse[1]

    beta0 = model.params[0]
    SE_beta0 = model.bse[0]

    print(beta0)
    print(fsr_theoretical)
    print(SE_beta0)
    t_c = (beta0-fsr_theoretical)/SE_beta0

    # Calculate the t-statistic
    t = beta1 / SE_beta1


    # print('*********************************************')
    # print(f'T Value: {np.abs(t)} and Critical Value: {critical_t}, for poly order {min_poly_order+i}')
    # if np.abs(t) > critical_t:
    #     print(f"Reject the null hypothesis, for poly order {min_poly_order+i}")
    # else:
    #     print(f"Fail to reject the null hypothesis, for poly order {min_poly_order+i}")
    # print('*********************************************')



    plt.axvline(t, color = colors[i+1], label = f'T-value: {t} at Poly Order {min_poly_order + i}')
    #plt.axvline(t_c, color = colors[i+1], label = f'C Value T-value: {t} at Poly Order {min_poly_order + i}')
plt.legend()

width = [1,1]
fig = plt.figure(figsize = (4,8))
LEGEND_SIZE = 8
gs = fig.add_gridspec(4, 2, hspace=0.7, wspace=0,width_ratios=width)
ax = gs.subplots(sharey=False, sharex=False)
fig.suptitle('T-statistic vs Poly Order (Testing Linearity)')
colors = plt.cm.rainbow(np.linspace(0, 1, len(array_of_coeffients)+1))

mid_point = (x_axis_peaks[1:] + x_axis_peaks[:-1]) / 2


m = (spacing[-1] - spacing[0])/ (mid_point[-1] - mid_point[0])
c = spacing[0] - m*mid_point[0]
inital_guess_straight = [m,c]

para,cov = curve_fit(straight_line,mid_point,spacing,inital_guess_straight)
perr = np.sqrt(np.diag(cov))
nstd = 3
para_up = para + nstd * perr
para_down = para - nstd * perr

fit_up = straight_line(mid_point, *para_up)
fit_dw = straight_line(mid_point, *para_down)
ax[0,0].fill_between(mid_point, fit_up, fit_dw, alpha=.25, label=str(nstd) + r'$\sigma$ Interval', color=colors[0])





print(f'T_STAT: {para[0]/np.sqrt(cov[0][0])}')

t_val = para[0]/np.sqrt(cov[0][0])
ax[0,1].axvline(t_val, label = f'T Statistic: {t_val}')

ax[0,0].plot(mid_point, straight_line(mid_point,*para), color=colors[0], label='Fitted line')

ax[0,0].scatter(mid_point,spacing, label = f'FP spacing', color = colors[0])
ax[0,0].legend(loc = 'best',prop={'size': LEGEND_SIZE})
ax[0,1].yaxis.tick_right()

ax[0,0].set_xlabel('Time Scale (s)')
ylabel = ax[0,0].set_ylabel('Time Scale (s)')


df = len(spacing) - 2

alpha_5 = 0.05
alpha_1 = 0.01
alpha_values = np.linspace(0,1,10000000)

critical_t_5_upper = stats.t.ppf(q=(1-(alpha_5/2)), df= df)
critical_t_1_upper = stats.t.ppf(q=(1 - (alpha_1/2)), df= df)
critical_t_5_lower = stats.t.ppf(q=alpha_5/2, df= df)
critical_t_1_lower = stats.t.ppf(q=alpha_1/2, df= df)

x_ = np.linspace(stats.t.ppf(0.0001, df), stats.t.ppf(0.9999, df), 10000)
y_ = stats.t.pdf(x_, df)


title = ax[0,0].set_title(r"$\bf{A.}$" + ' Original Data',loc='center', y=1.05)

offset = np.array([-0.15, -0.45])
title.set_position(ylabel.get_position() + offset)
title.set_rotation(90)

ax[0,1].set_title(r"Two-Tailed Hypothesis Testing using T-Statistic ($\nu$ = "+str(df)+r", $\alpha$ = 0.05)",loc='center', y=1.05)
letters = [chr(i) for i in range(ord('B'), ord('Z')+1)]

for i in range(0,4):
    ax[i,1].plot(x_, y_, color = 'black')

    # ax[i,1].axvline(critical_t_5_upper, color = 'purple')
    # ax[i,1].axvline(critical_t_5_lower,color = 'purple',)
    # plt.axvline(critical_t_1_upper, color = 'blue', label = f'Critical T-Value at 99% confidence (Two-Tailed) T: [{critical_t_1_upper}]')
    # plt.axvline(critical_t_1_lower, color = 'blue')
    ax[i,1].fill_between(x_,y_,0,where=(x_>=critical_t_5_lower) & (x_<=critical_t_5_upper),color='green', alpha=0.5)
    ax[i,1].fill_between(x_,y_,0,where=(x_>=min(x_)) & (x_<=critical_t_5_lower),color='red', alpha=0.5)
    ax[i,1].fill_between(x_,y_,0,where=(x_>=critical_t_5_upper) & (x_<=max(x_)),color='red',alpha=0.5)
    ax[i,1].set_xlabel('T-Statistic (No units)')
    ax[i,1].set_ylabel('P.D.F (No units)')
    ax[i,1].legend(loc = 'best',prop={'size': LEGEND_SIZE})
    ax[i,1].yaxis.tick_right()
    ax[i,1].yaxis.set_label_position("right")



for i in range(len(array_of_coeffients)):
    j = i + 1
    ax[j,0].set_xlabel('Relative Frequency (Hz)')
    ylabel = ax[j,0].set_ylabel('Relative Frequency (Hz)')

    title = ax[j,0].set_title(r"$\bf{" + str(letters[i]) + ".}$"+ f' Polyorder {i+min_poly_order}',loc='center', y=1.05)

    offset = np.array([-0.15, -0.45])
    title.set_position(ylabel.get_position() + offset)
    title.set_rotation(90)

    freqency_array = f(normalized_x_axis,*array_of_coeffients[i])
    freq_peaks = freqency_array[peaks]
    freq_peaks = freq_peaks[1:]
    mid_val = (freq_peaks[1:] + freq_peaks[:-1])/2
    spacing_freq = np.diff(freq_peaks)

    m = (spacing_freq[-1] - spacing_freq[0])/ (mid_val[-1] - mid_val[0])
    c = spacing_freq[0] - m*mid_val[0]
    inital_guess_straight = [m,c]

    para,cov = curve_fit(straight_line,mid_val,spacing_freq,inital_guess_straight)
    perr = np.sqrt(np.diag(cov))
    nstd = 3
    para_up = para + nstd * perr
    para_down = para - nstd * perr

    fit_up = straight_line(mid_val, *para_up)
    fit_dw = straight_line(mid_val, *para_down)
    ax[j,0].fill_between(mid_val, fit_up, fit_dw, alpha=.25, label=str(nstd) + r'$\sigma$ Interval', color=colors[j])

    ax[j,0].plot(mid_val, straight_line(mid_val,*para), color=colors[j], label='Fitted line')

    ax[j,0].scatter(mid_val,spacing_freq, label = f'FP spacing', color = colors[j])
    # ax[j,0].axhline(max(spacing_freq), color = colors[i])
    # ax[j,0].axhline(min(spacing_freq), color = colors[i])
    ax[j,0].legend(loc = 'best',prop={'size': LEGEND_SIZE})

    t_val = para[0]/np.sqrt(cov[0][0])
    ax[j,1].axvline(t_val, label = f'T Statistic: {t_val}')
    ax[j,1].yaxis.tick_right()
    ax[j,1].yaxis.set_label_position("right")
    ax[j,1].legend(loc = 'best',prop={'size': LEGEND_SIZE})



plt.legend(prop={'size': LEGEND_SIZE})
plt.show()