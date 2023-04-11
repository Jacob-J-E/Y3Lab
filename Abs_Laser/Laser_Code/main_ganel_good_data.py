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
max_poly_order = 2
min_poly_order = 2
if min_poly_order > max_poly_order:
    print('Make sure max poly order is higher or equal to min poly order')
    exit()
elif min_poly_order < 2:
    print('Please put a coeff greater than or equal to 2')
    exit()


R_1_R_2 = 0.995*0.995
F = (np.pi * (R_1_R_2)**(1/4))/(1-(R_1_R_2)**(1/2))
#F = (np.pi/2)*((np.arcsin((1-np.sqrt(R_1_R_2))/(2*(R_1_R_2))**(1/4)))**(-1))
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

directory = r"Abs_Laser\Data\21-03-2023\\"
data = pd.read_csv(directory+"Z0.CSV")
data_DB_free = pd.read_csv(directory+"ZB1.CSV")

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



# c1_B = c1_B[(x_axis > -0.012) & (x_axis < 0.02094)]
# c1 = c1[(x_axis > -0.012) & (x_axis < 0.02094)]
# c4 = c4[(x_axis > -0.012) & (x_axis < 0.02094)]
# c3 = c3[(x_axis > -0.012) & (x_axis < 0.02094)]
# x_axis = x_axis[(x_axis > -0.012) & (x_axis < 0.02094)]

# x_axis = x_axis[::-1]
# x_axis, c3, c4,c1,c1_B = zip(*sorted(zip(x_axis, c3, c4,c1,c1_B )))

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


peaks, _= find_peaks(c4, prominence = 0.6 )
c4_peaks = c4[peaks]
c4_peaks_= c4_peaks
x_axis_peaks = normalized_x_axis[peaks]
x_axis_peaks_ = x_axis_peaks
x_axis_peaks = x_axis_peaks[1:]
c4_peaks = c4_peaks[1:]
spacing = np.diff(x_axis_peaks)

plt.figure()
c1_B_ = c1_B - min(c1_B)
c1_B_= c1_B_/max(c1_B_)
plt.plot(normalized_x_axis,c1_B_)
plt.scatter((x_axis_peaks[1:] + x_axis_peaks[:-1])/2 ,spacing/max(spacing))

plt.figure()
plt.plot(normalized_x_axis,c4)
plt.scatter(x_axis_peaks_,c4_peaks_)
plt.show()

lorentzian_array = np.zeros(len(normalized_x_axis))
FP_lorenzian_x_axis_peaks = []
FP_lorenzian_x_axis_peaks_amplitude = []
for i in range(len(x_axis_peaks_)):
    inital_guess = [x_axis_peaks_[i], 0.00004,c4_peaks_[i], 0]  
    para, cov = curve_fit(lorentzian,normalized_x_axis, c4, inital_guess)
    y_data = lorentzian(normalized_x_axis,para[0], para[1], para[2], para[3])
    FP_lorenzian_x_axis_peaks.append(para[0])
    FP_lorenzian_x_axis_peaks_amplitude.append(max(y_data) - para[3])
    lorentzian_array += y_data - para[3]

FP_lorenzian_x_axis_peaks = np.array(FP_lorenzian_x_axis_peaks)
FP_lorenzian_x_axis_peaks_amplitude = np.array(FP_lorenzian_x_axis_peaks_amplitude)

plt.figure()
plt.plot(normalized_x_axis,lorentzian_array, label = 'lor')
plt.plot(normalized_x_axis,c4, label = 'data')
plt.show()

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



#Fitted FP Lorentzian (Original Dataset) Peak Spacings
#'Frequency Scaled Broadened Absorption Spectra'
#Frequency Scaled Hyperfine Structure
#Frequency Scaled Hyperfine Structure (Rb 87 Ground State F = 2 -> F=1,2,3)
def plot_functions(FITTED_OG_FP = False, FREQ_SCALE_FP = False, FREQ_SCALED_SPACING = False, T_STAT = False, OG_DATA_SPACING = False, FREQ_SCALED_GROUNDSTATE = False,FREQ_SCALED_HYPERFINE = False,FREQ_SCALED_1_HYPERFINE = False,FREQ_SCALED_2_HYPERFINE = False):
    if FITTED_OG_FP == True:
        plt.figure()
        domain_airy_modified = np.linspace(min(normalized_x_axis),max(normalized_x_axis), 100000)

        plt.title('Fitted FP using Modified Airy Function')
        for i in range(len(array_of_coeffients)):
            para = list(array_of_coeffients[i]) + list(b_values[i])
            plt.plot(domain_airy_modified,airy_modified_function(domain_airy_modified,*para), label =  f"Fitted Airy (f(x) poly order {min_poly_order + i})" )
        plt.plot(normalized_x_axis,lorentzian_array, label =  "Data" )
        plt.legend()
    if FREQ_SCALE_FP == True:
        plt.figure()
        plt.title('Frequency Scaled FP')
        for i in range(len(array_of_coeffients)):
            freqency_array = f(normalized_x_axis,*array_of_coeffients[i])
            plt.plot(freqency_array,lorentzian_array, label =  f"Frequency Scaling (poly order {min_poly_order + i})" )
        plt.xticks(np.arange(0, max(freqency_array)+fsr_theoretical, fsr_theoretical))
        plt.legend()
    if FREQ_SCALED_SPACING == True:
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

        if T_STAT == True:
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
            plt.axvline(t, color = colors[0], label = f'T-value: {t} (Original FP Spacing)')

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
            plt.axvline(critical_t_1_upper, color = 'blue', label = f'Critical T-Value at 99% confidence (Two-Tailed) T: [{critical_t_1_upper}]')
            plt.axvline(critical_t_1_lower, color = 'blue')
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
            plt.legend()

        if OG_DATA_SPACING == True:
            plt.figure()
            plt.title('(Original Dataset) Peak Spacings')
            mid_point = (x_axis_peaks[1:] + x_axis_peaks[:-1]) / 2
            plt.scatter(mid_point,spacing, label = 'Time scale')
            plt.legend()
        
        plt.show()
        if FREQ_SCALED_GROUNDSTATE == True:
            #Plotting the Broadened Absorption Spectra----------------------------------------------------------------
            plt.figure()
            plt.title('Frequency Scaled Broadened Absorption Spectra')
            for i in range(len(array_of_coeffients)):
                peaks_fine, _= find_peaks(-1*c1_B, distance=8000)
                c1_b_grouped_peaks =c1_B[peaks_fine]
                freqency_array = f(normalized_x_axis,*array_of_coeffients[i])
                freq_peaks = freqency_array[peaks_fine]

                centers_fine = [freq_peaks[1],freq_peaks[2],freq_peaks[4],freq_peaks[6]]
                amplitude_fine = [c1_b_grouped_peaks[1],c1_b_grouped_peaks[2],c1_b_grouped_peaks[4],c1_b_grouped_peaks[6]]

                Rb_85_Ground_State_Difference = (freq_peaks[4]-freq_peaks[2])/1e9
                Rb_87_Ground_State_Difference = (freq_peaks[6]-freq_peaks[1])/1e9

                Rb_85_Ground_State_Difference_THEORY = 3.0357324390
                Rb_87_Ground_State_Difference_THEORY = 6.834682610904290

                Rb_85_Ground_State_Difference_THEORY_PD = ((Rb_85_Ground_State_Difference_THEORY - Rb_85_Ground_State_Difference) / Rb_85_Ground_State_Difference_THEORY) * 100
                Rb_87_Ground_State_Difference_THEORY_PD = ((Rb_87_Ground_State_Difference_THEORY - Rb_87_Ground_State_Difference) / Rb_87_Ground_State_Difference_THEORY) * 100
            
                print('************************************')
                print(f'Poly Order: {min_poly_order + i}')
                print(f'Hyperfine Structure Rb 85 Ground State Difference: {Rb_85_Ground_State_Difference:.4g} MHz, Theory P.D: {Rb_85_Ground_State_Difference_THEORY_PD:.4g}%')
                print(f'Hyperfine Structure Rb 87 Ground State Difference: {Rb_87_Ground_State_Difference:.4g} MHz, Theory P.D: {Rb_87_Ground_State_Difference_THEORY_PD:.4g}%')
                print('****************************')

                plt.plot(freqency_array,c1_B, label = f'Data (poly order {min_poly_order + i})')
                plt.scatter(centers_fine,amplitude_fine, marker= 'x', label = f'Scatter (poly order {min_poly_order + i})')
            plt.legend()

            if FREQ_SCALED_HYPERFINE == True:
                #Plotting Hyperfine Structure ------------------------------------------------------
                plt.figure()
                plt.title('Frequency Scaled Hyperfine Structure')
                for i in range(len(array_of_coeffients)):
                    freqency_array = f(normalized_x_axis,*array_of_coeffients[i])
                    hyper_fine_structure = c1-c1_B
                    hyper_fine_structure = hyper_fine_structure - min(hyper_fine_structure)
                    hyper_fine_structure = hyper_fine_structure/max(hyper_fine_structure)
                    plt.plot(freqency_array,hyper_fine_structure,label = f'Data Hyperfine Structure (poly order {min_poly_order + i})')
                plt.legend()

            if FREQ_SCALED_1_HYPERFINE == True:
                plt.figure()
                plt.title('Frequency Scaled Hyperfine Structure (Rb 87 Ground State F = 2 -> F=1,2,3)')
                for i in range(len(array_of_coeffients)):
                    freqency_array = f(normalized_x_axis,*array_of_coeffients[i])
                    hyper_fine_structure = c1-c1_B

                    hyper_fine_structure = hyper_fine_structure[(freqency_array > 6.954e8) & (freqency_array < 1.4992e9)]
                    freqency_array = freqency_array[(freqency_array > 6.954e8) & (freqency_array < 1.4992e9)]

                    hyper_fine_structure = hyper_fine_structure - min(hyper_fine_structure)
                    hyper_fine_structure = hyper_fine_structure/max(hyper_fine_structure)

                    plt.plot(freqency_array,hyper_fine_structure,label = f'Data Hyperfine Structure (poly order {min_poly_order + i})')

                    peaks_fine, _= find_peaks(hyper_fine_structure, distance=400)
                    subtracted_peaks = hyper_fine_structure[peaks_fine]
                    freq_peaks = freqency_array[peaks_fine]

                    peaks_values = [subtracted_peaks[2],subtracted_peaks[3],subtracted_peaks[4],subtracted_peaks[5],subtracted_peaks[7],subtracted_peaks[1]]
                    freq_values = [freq_peaks[2],freq_peaks[3],freq_peaks[4],freq_peaks[5],freq_peaks[7],freq_peaks[1]]
                    peaks_values_inital_guess = np.array(peaks_values) * 1e7
                    width = 9e6
                    #initial_guess = [freq_values[0],width,peaks_values_inital_guess[0],freq_values[1],width,peaks_values_inital_guess[1],freq_values[2],width,peaks_values_inital_guess[2],freq_values[3],width,peaks_values_inital_guess[3],freq_values[4],width,peaks_values_inital_guess[4], 1e-30,1e-30,0.03]
                    #f1,w1,a1,c1,f2,w2,a2,c2,f3,w3,a3,c3,f4,w4,a4,c4,f5,w5,a5,c5,c

                    m = (hyper_fine_structure[-1] - hyper_fine_structure[0])/ (freqency_array[-1] - freqency_array[0])
                    c = hyper_fine_structure[0] - m*freqency_array[0]
                    maybe_peak = 2*freq_values[0] - freq_values[1]
                    initial_guess = [freq_values[0],width+8e6,peaks_values_inital_guess[0],freq_values[1],width+30e6,peaks_values_inital_guess[1]+0.8e6,freq_values[2],width,peaks_values_inital_guess[2],freq_values[3],width,peaks_values_inital_guess[3],freq_values[4],width,peaks_values_inital_guess[4],maybe_peak,width,peaks_values_inital_guess[5],m,c]
                    params, cov = curve_fit(five_lor_x_update,freqency_array,hyper_fine_structure,initial_guess)
                    plt.plot(freqency_array,five_lor_x_update(freqency_array,*params),color='black')
                    # plt.plot(freqency_array,five_lor_x_update(freqency_array,*params) - np.array((params[-3]*freqency_array**2 + params[-2]*freqency_array - params[-1])),color='red')
                    plt.plot(freqency_array,five_lor_x_update(freqency_array,*initial_guess),label = 'Inital Guess', color = 'orange')
                    plt.plot(freqency_array,hyper_fine_structure - straight_line(freqency_array,initial_guess[-2],initial_guess[-1]), label = 'SHIFTED', color = 'purple')
                    print((freq_values[4] - freq_values[1])/1e6)
                    print((freq_values[1] - maybe_peak)/1e6)


                    Rb_87_2_to_3_THEORY = 266.6500
                    Rb_87_1_to_2_THEORY = 156.9470

                    direct_Rb_87_2_to_3 = (params[12] - params[3])/ 1E6
                    direct_Rb_87_2_to_3_Difference_THEORY_PD = 100* ((direct_Rb_87_2_to_3 - Rb_87_2_to_3_THEORY) / Rb_87_2_to_3_THEORY)

                    c1_2 = params[0]
                    c1_3 = params[6]
                    c2_3 = params[9]

                    # print(indirect_Rb_87_g2_to_1, 'MONEY')
                    # print(indirect_Rb_87_g2_to_3, 'MONEY')
                    # print(indirect_Rb_87_g2_to_2, 'MONEY')

                    indirect_Rb_87_1_to_2 = 2*(c2_3 - c1_3)/1e6
                    indirect_Rb_87_2_to_3 = 2*(c1_3 - c1_2)/1e6

                    indirect_Rb_87_1_to_2_Difference_THEORY_PD = 100*((indirect_Rb_87_1_to_2 - Rb_87_1_to_2_THEORY)/Rb_87_1_to_2_THEORY)
                    indirect_Rb_87_2_to_3_Difference_THEORY_PD = 100*((indirect_Rb_87_2_to_3 - Rb_87_2_to_3_THEORY)/Rb_87_2_to_3_THEORY)

                    print('************************************')
                    print(f'Poly Order: {min_poly_order + i}')
                    print(f'(Direct Method) Hyperfine Structure Rb 87 excited - F=2 -> 3 : {direct_Rb_87_2_to_3:.4g} MHz, Theory P.D: {direct_Rb_87_2_to_3_Difference_THEORY_PD:.4g}%')
                    print(f'(Indirect Method) Hyperfine Structure Rb 87 excited - F=2 -> 3 : {indirect_Rb_87_2_to_3:.4g} MHz, Theory P.D: {indirect_Rb_87_2_to_3_Difference_THEORY_PD:.4g}%')
                    print(f'(Indirect Method) Hyperfine Structure Rb 87 excited - F=1 -> 2 : {indirect_Rb_87_1_to_2:.4g} MHz, Theory P.D: {indirect_Rb_87_1_to_2_Difference_THEORY_PD:.4g}%')
                    print('************************************')
                    plt.scatter(freq_peaks,subtracted_peaks, marker= 'x',label = f'Peak Points (poly order {min_poly_order + i})')
                plt.legend()

            if FREQ_SCALED_2_HYPERFINE == True:
                plt.figure()
                plt.title('Frequency Scaled Hyperfine Structure (Rb 85 Ground State F = 3 -> F=2,3,4)')
                for i in range(len(array_of_coeffients)):
                    freqency_array = f(normalized_x_axis,*array_of_coeffients[i])
                    hyper_fine_structure = c1-c1_B

                    hyper_fine_structure = hyper_fine_structure[(freqency_array > 1.9e9) & (freqency_array < 2.6e9)]
                    freqency_array = freqency_array[(freqency_array > 1.9e9) & (freqency_array < 2.6e9)]

                    hyper_fine_structure = hyper_fine_structure - min(hyper_fine_structure)
                    hyper_fine_structure = hyper_fine_structure/max(hyper_fine_structure)

                    plt.plot(freqency_array,hyper_fine_structure,label = f'Data Hypserfine Structure (poly order {min_poly_order + i})')

                    peaks_fine, _= find_peaks(hyper_fine_structure, distance=150)
                    subtracted_peaks = hyper_fine_structure[peaks_fine]
                    freq_peaks = freqency_array[peaks_fine]

                    peaks_values = [subtracted_peaks[10],subtracted_peaks[13],subtracted_peaks[14],subtracted_peaks[16]]
                    freq_values = [freq_peaks[10],freq_peaks[13],freq_peaks[14],freq_peaks[16]]
                    peaks_values_inital_guess = np.array(peaks_values)
                    m = (hyper_fine_structure[-1] - hyper_fine_structure[0])/ (freqency_array[-1] - freqency_array[0])
                    c = hyper_fine_structure[0] - m*freqency_array[0]
                    width = 1.2e7
                    #initial_guess = [freq_values[0]-0.005e9,width+30e6,peaks_values_inital_guess[0]+3e6,freq_values[1],width+20e6,peaks_values_inital_guess[1]+30e6,freq_values[2],width,peaks_values_inital_guess[2]+3e6,freq_values[3],width,peaks_values_inital_guess[3],m,c]
                    initial_guess = [freq_values[0],width+100e6,peaks_values_inital_guess[0],freq_values[1],width,peaks_values_inital_guess[1],freq_values[2],width,peaks_values_inital_guess[2],freq_values[3],width,peaks_values_inital_guess[3],m,c]
                    params, cov = curve_fit(four_lor_x_update,freqency_array,hyper_fine_structure,initial_guess)
                    # initial_guess = params
                    # params, cov = curve_fit(four_lor_x_update,freqency_array,hyper_fine_structure,initial_guess)
                    plt.plot(freqency_array,four_lor_x_update(freqency_array,*params),color='black')
                    plt.plot(freqency_array,four_lor_x_update(freqency_array,*initial_guess),label = 'Inital Guess', color = 'orange')
                    plt.plot(freqency_array,hyper_fine_structure - straight_line(freqency_array,initial_guess[-2],initial_guess[-1]), label = 'SHIFTED', color = 'purple')
                    plt.scatter(freq_values,peaks_values, marker= 'x',label = f'Peak Points (poly order {min_poly_order + i})')
                    plt.scatter(freq_peaks,subtracted_peaks, marker= 'x', color = 'red',label = f'Peak Points (poly order {min_poly_order + i})')
                    Rb_85_2_to_3_THEORY = 63.401
                    Rb_85_3_to_4_THEORY = 120.640

                    c_2_4 = params[3]
                    c_3_4 = params[6]
                    v_4 = params[9]
                    

                    indirect_Rb_85_2_to_3 = 2*(c_3_4 - c_2_4)/1e6
                    indirect_Rb_85_3_to_4 = 2*(v_4 - c_3_4)/1e6

                    indirect_Rb_85_2_to_3_Difference_THEORY_PD = 100*((indirect_Rb_85_2_to_3 - Rb_85_2_to_3_THEORY)/Rb_85_2_to_3_THEORY)
                    indirect_Rb_85_3_to_4_Difference_THEORY_PD = 100*((indirect_Rb_85_3_to_4 - Rb_85_3_to_4_THEORY)/Rb_85_3_to_4_THEORY)

                    print('************************************')
                    print(f'Poly Order: {min_poly_order + i}')
                    print(f'(Indirect Method) Hyperfine Structure Rb 85 excited - F=2 -> 3 : {indirect_Rb_85_2_to_3:.4g} MHz, Theory P.D: {indirect_Rb_85_2_to_3_Difference_THEORY_PD:.4g}%')
                    print(f'(Indirect Method) Hyperfine Structure Rb 85 excited - F=3 -> 4 : {indirect_Rb_85_3_to_4:.4g} MHz, Theory P.D: {indirect_Rb_85_3_to_4_Difference_THEORY_PD:.4g}%')
                    print('************************************')

                plt.legend()

            plt.show()



plot_functions(FITTED_OG_FP = True, FREQ_SCALE_FP = True, FREQ_SCALED_SPACING = True, T_STAT = True, OG_DATA_SPACING = True, FREQ_SCALED_GROUNDSTATE = True,FREQ_SCALED_HYPERFINE = True,FREQ_SCALED_1_HYPERFINE = True,FREQ_SCALED_2_HYPERFINE = True)



