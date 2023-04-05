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
#/Abs_Laser/freq_jank.py
# C:\Users\Ellre\Desktop\Physics Degree\LAB\Year 3 Lab\Y3Lab\Abs_Laser\freq_jank.py

def tikhonov_objective(recovered_signal, convolved_signal, gaussian_kernel, alpha):
    residual = convolved_signal - convolve1d(recovered_signal, gaussian_kernel, mode='reflect')
    regularization_term = alpha * np.sum(recovered_signal ** 2)
    return np.sum(residual ** 2) + regularization_term

def Gauss(x,A,mu,sigma):
    return A * np.exp(-(x-mu)**2/(2*sigma**2))

def Gauss_updated(x,A,mu,sigma,c):
    return A * np.exp(-(x-mu)**2/(2*sigma**2)) + c

def line(x,m,c):
    return m*x + c


def four_gauss(x,A1,mu1,sigma1,A2,mu2,sigma2,A3,mu3,sigma3, A4,mu4,sigma4, m, c):
    return Gauss(x,A1,mu1,sigma1) + Gauss(x,A2,mu2,sigma2) + Gauss(x,A3,mu3,sigma3) + Gauss(x,A4,mu4,sigma4) +line(x,m,c)

def wiener_filter(img, kernal, k):
    print("Sum kernal",np.sum(kernal))
    kernal = kernal/np.sum(kernal)
    dummy = np.copy(img)
    dummy = np.fft.rfft(img)
    kernal = np.fft.rfft(kernal)
    kernal = np.conj(kernal) / (np.abs(kernal)**2  + k)
    dummy = dummy * kernal
    dummy = np.abs(np.fft.irfft(dummy))
    return dummy

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

def voigt(x, center, sigma, gamma, amplitude, c):
    """
    Calculate the Voigt function for the given x values.

    Parameters
    ----------
    x : array-like
        Input x values to calculate the Voigt function.
    center : float
        The center of the Voigt function.
    sigma : float
        The standard deviation of the Gaussian component.
    gamma : float
        The half-width at half-maximum of the Lorentzian component.
    amplitude : float
        The amplitude of the Voigt function.

    Returns
    -------
    y : array-like
        The Voigt function values corresponding to the input x values.
    """
    # voight = voigt_profile(x - center, sigma, gamma) 
    y = amplitude * voigt_profile(x - center, sigma, gamma)  + c
    return y



"""
Rb Data
"""
# lines_87_2 = [3.842276916e14, 3.842278486e14, 3.842281152e14]
# lines_87_1 = [3.842344541e14, 3.842345263e14, 3.842346832e14]

# lines_85_3 = [3.842290576e14, 3.842291211e14, 3.842292417e14]
# lines_85_2 = [3.84232064e14, 3.842320934e14, 3.842321568e14]

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




#loading in data
'''
loading in data
'''
# data = pd.read_csv(r"Abs_Laser\Data\14-03-2023\DUB03B.CSV")
# data_DB_free = pd.read_csv(r"Abs_Laser\Data\14-03-2023\DUB03.CSV")

data = pd.read_csv(r"Abs_Laser\Data\10-03-2023\NEW1B.CSV")
data_DB_free = pd.read_csv(r"Abs_Laser\Data\10-03-2023\NEW1.CSV")

# data_DB_free = pd.read_csv(r"Abs_Laser\Data\21-03-2023\Z0.CSV")
# data = pd.read_csv(r"Abs_Laser\Data\21-03-2023\ZB0.CSV")

x_axis = data_DB_free['in s']
# x_axis = np.array(x_axis)-min(x_axis)
c1 = data_DB_free['C1 in V']           
c2 = data_DB_free['C2 in V'] 
c3 = data_DB_free['C3 in V']
c4 = data_DB_free['C4 in V']

c1_B = data['C1 in V']

c1_B = c1_B/max(c1_B)*max(c1)
x2 = data['in s']
#------------------------------------------------------------------------------------------
c1_B = np.array(c1_B)
c1 = np.array(c1)
x_axis = np.array(x_axis)

c1_B = c1_B[(x_axis > -0.01749) & (x_axis < 0.01573)]
c1 = c1[(x_axis > -0.01749) & (x_axis < 0.01573)]
c4 = c4[(x_axis > -0.01749) & (x_axis < 0.01573)]
c3 = c3[(x_axis > -0.01749) & (x_axis < 0.01573)]
x_axis = x_axis[(x_axis > -0.01749) & (x_axis < 0.01573)]

x_axis = np.array(x_axis)
# x_axis = 1 / x_axis
x_axis = x_axis[::-1]

x_axis, c3, c4,c1,c1_B = zip(*sorted(zip(x_axis, c3, c4,c1,c1_B )))


# c3 = c3[::-1]
# c4 = c4[::-1]
# c1 = c1[::-1]
# c1_B = c1_B[::-1]

# x_axis, c3, c4,c1,c1_B = zip(*sorted(zip(x_axis, c3, c4,c1,c1_B )))

x_axis = np.array(x_axis)
c3 = np.array(c3)
c4 = np.array(c4)
c1 = np.array(c1)
c1_B = np.array(c1_B)

plt.figure()
plt.plot(x_axis,c4)
plt.plot(x_axis,c3)
plt.plot(x_axis,c1)
plt.plot(x_axis,c1_B)


plt.figure()
plt.plot(x_axis,c1-c1_B/max(c1_B)*max(c1))
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
    print(para)
    plt.plot(x_axis_linspace,y_data,color='black')
    center_array.append(para[0])
    amplitude_array.append(max(y_data))
    std_array.append((4*para[1])/(2*np.sqrt(2*np.log(2))))
    cov_array.append(np.sqrt(cov[0][0]))


plt.scatter(center_array,amplitude_array,marker = 'x',color='green', label='Fitted Lorentzian')
plt.errorbar(center_array,amplitude_array,xerr=1*np.array(std_array),ls='None',color='green',capsize=5,label= 'Fitted Lorentzian')
# FP = np.array(c4)
# FP_sav = savgol_filter(FP,window_length=151,polyorder=3)
# max_ind = argrelextrema(FP_sav,np.greater)
# peak_y = FP_sav[max_ind[0]]
# peak_x = np.array(x_axis[max_ind[0]][peak_y > -0.056])
# peak_y = peak_y[peak_y > -0.056]
# spacing = np.diff(peak_x)

FP = np.array(c4)
peak_y = np.array(amplitude_array)
peak_x = np.array(center_array)
spacing = np.diff(peak_x)
#------------------------------------------------------------------------------------------



'''
remove extreme peak values
'''
q3, q1 = np.percentile(spacing, [75 ,25])
iqr = q3 - q1
peak_x_spliced = peak_x[:-1][(spacing < (q3+ 3*iqr)) & (spacing > (q1- 3*iqr)) ]
spacing_spliced = spacing[(spacing < (q3+ 3*iqr)) & (spacing > (q1- 3*iqr)) ]

print('spacing_spliced av',np.average(spacing_spliced))
print('spacing_spliced std',np.std(spacing_spliced))

# points = [[peak_x_spliced[i],spacing_spliced[i]] for i in range(len(spacing_spliced))]
# points = np.array(points)
# eps_size = max(spacing_spliced) - min(spacing_spliced)/15
# dbscan = DBSCAN(eps=eps_size, min_samples=2)
# clusters = dbscan.fit_predict(points)

# print(np.diff(spacing_spliced))
# print(np.average(np.diff(spacing_spliced)))
#------------------------------------------------------------------------------------------

'''
Grouping peak values
'''
difference_of_spacing = np.diff(spacing_spliced)
q3_ds, q1_ds = np.percentile(difference_of_spacing, [90 ,15])
iqr_ds = q3_ds - q1_ds
threshold_distance = iqr_ds

print(f'threshold_distance: {threshold_distance}')

array_spacing = []
array_x_peaks = []
group_spacing = []
group_x_peaks = []
for i in range(len(difference_of_spacing)+1):
    if(i == len(difference_of_spacing)):
        dif = spacing_spliced[i] - spacing_spliced[i-1]
        if dif > threshold_distance:
            array_spacing.append([spacing_spliced[i]])
            array_x_peaks.append([peak_x_spliced[i]])
            group_spacing = []
            group_x_peaks = []
        else:
            group_spacing.append(spacing_spliced[i])
            group_x_peaks.append(peak_x_spliced[i])
            array_spacing.append(group_spacing)
            array_x_peaks.append(group_x_peaks)
    else:
        if abs(difference_of_spacing[i]) > threshold_distance:
            group_spacing.append(spacing_spliced[i])
            group_x_peaks.append(peak_x_spliced[i])
            array_spacing.append(group_spacing)
            array_x_peaks.append(group_x_peaks)
            group_spacing = []
            group_x_peaks = []
        else:
            group_spacing.append(spacing_spliced[i])
            group_x_peaks.append(peak_x_spliced[i])


# print(array_spacing, 'array_spacing')
# print(array_x_peaks, 'array_x_peaks')

#------------------------------------------------------------------------------------------


'''
interpolating grouped values
'''
inter_y = []
inter_x = []
av_points = 2
for i in range(len(array_spacing)):
    xvals = []
    if(len(array_x_peaks[i]) > av_points):
       print("Option A")
       separation = int(len(array_x_peaks[i])/av_points)
       xvals = np.linspace(min(array_x_peaks[i]), max(array_x_peaks[i]), separation)
       yinterp = np.interp(xvals, array_x_peaks[i], array_spacing[i])
       inter_y.append(yinterp)
       inter_x.append(xvals)
    elif len(array_x_peaks[i]) == 2:
        print("Option B")
        xvals = np.array([min(array_x_peaks[i]), max(array_x_peaks[i])])
        print(f'array_spacing {array_spacing}')
        print(f'xvals {xvals}')
        print(f'array_x_peaks {array_x_peaks}')
        g = np.divide((array_spacing[i][-1] - array_spacing[i][0]),(array_x_peaks[i][-1] - array_x_peaks[i][0]))
        #g = (array_spacing[i][-1] - array_spacing[i][0])/(array_x_peaks[i][-1] - array_x_peaks[i][0])
        c = array_spacing[i][-1]-g*array_x_peaks[i][-1]
        y1 = g*min(array_x_peaks[i])+c
        y2 = g*max(array_x_peaks[i])+c
        yinterp = np.array([y1,y2])
        inter_y.append(yinterp)
        inter_x.append(xvals)

    # average = np.average(array_spacing[i])
    # range = max(array_spacing[i]) - min(array_spacing[i])
    # separation = int(range/(average))
    # if separation == 0:
    #     if(len(array_x_peaks[i]) >= 2):
    #        separation = int(len(array_x_peaks[i])/2)
    #     else: 
    #         separation = 1

    # print(separation')

    # yinterp = np.interp(xvals, array_x_peaks[i], array_spacing[i])
    # inter_y.append(yinterp)
    # inter_x.append(xvals)

print(array_spacing, 'array_spacing')
print(array_x_peaks, 'array_x_peaks')
print(inter_y, 'inty')
print(inter_x, 'intx')
#------------------------------------------------------------------------------------------
'''
Using interpolating grouped values to further separate peakvalues
'''
grouped_spacing_inter = []
grouped_x_peaks_inter = []
for i in range(len(inter_x)):
    for j in range(len(inter_x[i])-1):
        grouped_spacing_inter.append(spacing_spliced[(peak_x_spliced <= inter_x[i][j+1]) & (peak_x_spliced >= inter_x[i][j])])
        grouped_x_peaks_inter.append(peak_x_spliced[(peak_x_spliced <= inter_x[i][j+1]) & (peak_x_spliced >= inter_x[i][j])])
#------------------------------------------------------------------------------------------
'''
Calculates the average FSR for each group
'''
FSR_array = []
for i in range(len(grouped_spacing_inter)):
    FSR = np.average(grouped_spacing_inter[i])
    FSR_array.append([FSR,min(grouped_x_peaks_inter[i]),max(grouped_x_peaks_inter[i])])

print(FSR_array)
#------------------------------------------------------------------------------------------
'''
Groups the x-axis in between interpolated points. if it doesn't lie inbetween a point, it goes to nearest group.
'''

x_axis_grouping = []
j = 0
groupings = []
x_axis = np.array(x_axis)
#print(x_axis[-1])
for i in range(len(x_axis)):
    if j == len(FSR_array)-1:
        groupings.append(x_axis[i])
        if i == len(x_axis)-1:
            x_axis_grouping.append(groupings)
            break
    else:
        if abs(FSR_array[j][2] - x_axis[i]) < abs(FSR_array[j+1][1] - x_axis[i]):
                groupings.append(x_axis[i])
        else:
            groupings.append(x_axis[i])
            x_axis_grouping.append(groupings)
            j = j + 1
            groupings = []
#print(x_axis_grouping[-1][-1])
#-----------------------------------------------------------------

'''
Calculates scaling factor for time to freq
'''
scaling = []
FP_length = 2*19.5e-2
# FP_length = 2 * 19.5e-2
# FP_length = 2 * 20e-2


for i in range(len(FSR_array)):
    scaling.append(((3e8/(2*FP_length))/FSR_array[i][0]))

#-----------------------------------------------------------------

# print(scaling)
'''
shifts x axis to freq groupings
'''

# print(len(x_axis_grouping))
# print(len(scaling))

print(f'x_axis_grouping length {len(x_axis_grouping)}')
print(f'scaling length {len(scaling)}')
freq = [] 
for i in range(len(scaling)):
    freq.append(np.array(x_axis_grouping[i])*scaling[i])

#-----------------------------------------------------------------

c1_grouped = []
c1_b_grouped = []
k= 0
for i in range(len(x_axis_grouping)):
    g = []
    g_2 = []
    for j in range(len(x_axis_grouping[i])):
        g.append(c1[k])
        g_2.append(c1_B[k])
        k = k + 1
    c1_grouped.append(g)
    c1_b_grouped.append(g_2)
    

freq_flatten = []
c1_b_grouped_flatten = []
for i in range(len(freq)):
    freq_flatten.append(freq[i].tolist())
    c1_b_grouped_flatten.append(c1_b_grouped[i])

freq_flatten = list(chain.from_iterable(freq_flatten))
c1_b_grouped_flatten = list(chain.from_iterable(c1_b_grouped_flatten))

c1_b_grouped_flatten = np.array(c1_b_grouped_flatten)
freq_flatten = np.array(freq_flatten)
peaks_fine, _= find_peaks(-c1_b_grouped_flatten, distance=8000)
c1_b_grouped_peaks =c1_b_grouped_flatten[peaks_fine]
freq_peaks = freq_flatten[peaks_fine]
print(f'freq_peaks {freq_peaks}')
print(f'c1_b_grouped_peaks {c1_b_grouped_peaks}')

centers_fine = [freq_peaks[1],freq_peaks[2],freq_peaks[4],freq_peaks[6]]
amplitude_fine = [c1_b_grouped_peaks[1],c1_b_grouped_peaks[2],c1_b_grouped_peaks[4],c1_b_grouped_peaks[6]]

freq_linspace =np.linspace(freq_flatten[0],freq_flatten[-1], 1000000)
# for i in range(len(centers_fine)):
#     #inital_guess = [centers_fine[i], 1e4,0,amplitude_fine[i],0]
#     inital_guess = [amplitude_fine[i],centers_fine[i], 1e4,0]
#     try:
#         #bounds= ((-np.inf,0,0,-np.inf,np.inf),(np.inf,np.inf,np.inf,np.inf,np.inf))
#         para, cov = curve_fit(Gauss_updated, freq_flatten, c1_b_grouped_flatten, inital_guess)
#         print(f'para_voigt {para}')
#         plt.plot(freq_linspace,Gauss_updated(freq_linspace,para[0], para[1], para[2], para[3]))
#     except:
#         print("An exception occurred")
g = np.divide((c1_b_grouped_flatten[-1] - c1_b_grouped_flatten[0]),(freq_flatten[-1] - freq_flatten[0]))
c = c1_b_grouped_flatten[-1]-freq_flatten[-1]
inital_guess = [amplitude_fine[0],centers_fine[0],1e5,amplitude_fine[1],centers_fine[1],1e6,amplitude_fine[2],centers_fine[2],1e6,amplitude_fine[3],centers_fine[3],4e8, g,c]
para, cov = curve_fit(four_gauss, freq_flatten, c1_b_grouped_flatten, inital_guess)
para_updated = list([freq_linspace]) + para.tolist()
para_updated = np.array(para_updated)
print(f'para for gaussian {para}')
print("*************************************")
print(f'spacing 1 {(para[7] - para[4])/1e9} +/- {np.sqrt(para[8]**2 + para[5]**2)/1e9}')
print(f'spacing 2 {(para[10]- para[1])/1e9} +/- {np.sqrt(para[11]**2 + para[2]**2)/1e9}')
print("*************************************")
print(f'Freq 1 {(para[1])/1e9} +/- {(para[2])/1e9}')
print(f'Freq 1 {(para[4])/1e9} +/- {(para[5])/1e9}')
print(f'Freq 1 {(para[7])/1e9} +/- {(para[8])/1e9}')
print(f'Freq 1 {(para[10])/1e9} +/- {(para[11])/1e9}')
print("*************************************")
plt.figure()
plt.plot(freq_linspace,four_gauss(*para_updated))
plt.plot(freq_flatten,c1_b_grouped_flatten)
plt.scatter(freq_peaks,c1_b_grouped_peaks)


# offset = (384.230406373e12-1.7708439228e9) - (-4.9668e9)
# offset = (384.230406373e12-4.271676631815181e9) - para[1]
# offset = (384.230406373e12-2.563005979089109e9) - para[1]
offset = (384.230406373e12-2.563005979089109e9) - freq_peaks[1]# +0.1304e9

# offset = (384.230406373e12) - para[1]


# offset=0
freq_shifted = []
for  i in range(len(freq)):
    freq_shifted.append(freq[i]+offset)
freq_shifted =  np.array(freq_shifted)
# print(len(freq_flatten))
# print(len(x_axis))
# plt.plot(x_axis,c1)s
# plt.plot(x_axis,c4)
# plt.plot(x_axis,c1_B)
plt.figure()
plt.plot(x_axis,c4)
plt.scatter(peak_x,peak_y)
plt.title('WOAH')
plt.figure()
plt.scatter(peak_x[:-1],spacing)
plt.scatter(peak_x_spliced,spacing_spliced)
plt.figure()
# for cluster in np.unique(clusters):
#     plt.scatter(points[clusters == cluster, 0], points[clusters == cluster, 1], label=f"Cluster {cluster}")
for i in range(len(inter_y)):
    plt.scatter(array_x_peaks[i],array_spacing[i], label = f'cluster {i}' )
    plt.plot(inter_x[i], inter_y[i], '-x', label = f'cluster inter {i}')
plt.figure()
for i in range(len(grouped_x_peaks_inter)):
    plt.scatter(grouped_x_peaks_inter[i],grouped_spacing_inter[i])

for i in range(len(x_axis_grouping)):
    y_array = [0.0004 for i in range(len(x_axis_grouping[i]))]
    plt.scatter(x_axis_grouping[i],y_array)

# plt.figure()

#def voigt(x, center, sigma, gamma, amplitude, c):
#voigt(x, center, sigma, gamma, amplitude, c):
# for i in range(len(freq)):
#     c1_b_grouped_np = np.array(c1_b_grouped[i])
#     freq_np = np.array(freq[i])
#     peaks_fine, _= find_peaks(-c1_b_grouped_np, distance=10000)
#     c1_b_grouped_peaks =c1_b_grouped_np[peaks_fine]
#     freq_peaks = freq_np[peaks_fine]
#     plt.plot(freq_np,c1_grouped[i])
#     plt.plot(freq_np,c1_b_grouped_np)
#     plt.scatter(freq_peaks,c1_b_grouped_peaks)

#Gauss_updated(x,A,mu,sigma,c):
# peaks_fine, _= find_peaks(-c1_b_grouped_flatten, distance=8000)
# c1_b_grouped_peaks =c1_b_grouped_flatten[peaks_fine]
# freq_peaks = freq_flatten[peaks_fine]
# print(f'freq_peaks {freq_peaks}')
# print(f'c1_b_grouped_peaks {c1_b_grouped_peaks}')

# centers_fine = [freq_peaks[1],freq_peaks[2],freq_peaks[4],freq_peaks[6]]
# amplitude_fine = [c1_b_grouped_peaks[1],c1_b_grouped_peaks[2],c1_b_grouped_peaks[4],c1_b_grouped_peaks[6]]

# freq_linspace =np.linspace(freq_flatten[0],freq_flatten[-1], 1000000)
# # for i in range(len(centers_fine)):
# #     #inital_guess = [centers_fine[i], 1e4,0,amplitude_fine[i],0]
# #     inital_guess = [amplitude_fine[i],centers_fine[i], 1e4,0]
# #     try:
# #         #bounds= ((-np.inf,0,0,-np.inf,np.inf),(np.inf,np.inf,np.inf,np.inf,np.inf))
# #         para, cov = curve_fit(Gauss_updated, freq_flatten, c1_b_grouped_flatten, inital_guess)
# #         print(f'para_voigt {para}')
# #         plt.plot(freq_linspace,Gauss_updated(freq_linspace,para[0], para[1], para[2], para[3]))
# #     except:
# #         print("An exception occurred")
# g = np.divide((c1_b_grouped_flatten[-1] - c1_b_grouped_flatten[0]),(freq_flatten[-1] - freq_flatten[0]))
# c = c1_b_grouped_flatten[-1]-freq_flatten[-1]
# inital_guess = [amplitude_fine[0],centers_fine[0],1e5,amplitude_fine[1],centers_fine[1],1e6,amplitude_fine[2],centers_fine[2],1e6,amplitude_fine[3],centers_fine[3],4e8, g,c]
# para, cov = curve_fit(four_gauss, freq_flatten, c1_b_grouped_flatten, inital_guess)
# para_updated = list([freq_linspace]) + para.tolist()
# para_updated = np.array(para_updated)
# print(f'para for gaussian {para}')
# print("*************************************")
# print(f'spacing 1 {(para[7] - para[4])/1e9} +/- {np.sqrt(para[8]**2 + para[5]**2)/1e9}')
# print(f'spacing 2 {(para[10]- para[1])/1e9} +/- {np.sqrt(para[11]**2 + para[2]**2)/1e9}')
# print("*************************************")
# print(f'Freq 1 {(para[1])/1e9} +/- {(para[2])/1e9}')
# print(f'Freq 1 {(para[4])/1e9} +/- {(para[5])/1e9}')
# print(f'Freq 1 {(para[7])/1e9} +/- {(para[8])/1e9}')
# print(f'Freq 1 {(para[10])/1e9} +/- {(para[11])/1e9}')
# print("*************************************")

# plt.plot(freq_linspace,four_gauss(*para_updated))
# plt.plot(freq_flatten,c1_b_grouped_flatten)
# plt.scatter(freq_peaks,c1_b_grouped_peaks)
# plt.figure()
# plt.scatter(freq_flatten,c1)
plt.figure()
for i in range(len(freq)):
    plt.plot(freq[i],np.array(c1_grouped[i])-np.array(c1_b_grouped[i])+0.2,alpha=0.5)
plt.figure()
# sigma = 5e6
# sigma = 8e6
# sigma = 10/0.00002
offset = (3.84228115203221e14) - (-3.8400e9)
for i in range(len(freq)):
    plt.plot(freq[i] + offset,np.array(c1_grouped[i])-np.array(c1_b_grouped[i])+0.2,alpha=0.5)

for i in range(0,3):    
    plt.axvline(lines_85_2[i])
    plt.axvline(lines_85_3[i])
    plt.axvline(lines_87_2[i])
    plt.axvline(lines_87_1[i])
    plt.axvline(cross_lines_85_2[i],color='red')
    plt.axvline(cross_lines_85_3[i],color='red')
    plt.axvline(cross_lines_87_2[i],color='red')
    plt.axvline(cross_lines_87_1[i],color='red')

# for i in range(len(freq)):
#     print(i)
#     subtracted = np.array(c1_grouped[i])-np.array(c1_b_grouped[i])
#     blur = Gauss(freq[i],A=max(subtracted)/2,mu=np.mean(freq[i]),sigma=sigma)
#     if np.sum(blur) != 0:
#         blur_copy = blur.copy()
#         new_data = wiener_filter(subtracted,blur_copy,k=0.01)
        
#         min_splice = min(len(freq[i]),len(new_data))
#         plt.plot(freq[i][:min_splice]+ offset, new_data[:min_splice])
# plt.figure()
k_values = np.linspace(0.005,0.02,50)
s_values = np.linspace(5556000,1e8,50)
# print('Generating Data...')
# with alive_bar(len(k_values)*len(s_values)) as bar:
#     for s in s_values:
#         for k in k_values:
#                 # subtracted = np.array(c1_grouped[0])-np.array(c1_b_grouped[0])
#                 # blur = Gauss(freq[0],A=max(subtracted)/2,mu=np.mean(freq[0]),sigma=s)
#                 # if np.sum(blur) != 0:
#                 #     blur_copy = blur.copy()
#                 #     new_data = wiener_filter(subtracted,blur_copy,k=k)
                    
#                 #     min_splice = min(len(freq[0]),len(new_data))
#                 #     plt.figure()
#                 #     plt.plot(freq[0][:min_splice]+ offset, new_data[:min_splice])
#                 #     path = r'Abs_Laser/Images/Zero/'
#                 #     name = 'k='+str(k)+'sig='+str(s)+".png"
#                 #     plt.title(name)
#                 #     plt.savefig(fname = path+name, dpi = 200)

#                 subtracted = np.array(c1_grouped[1])-np.array(c1_b_grouped[1])
#                 blur = Gauss(freq[1],A=max(subtracted)/2,mu=np.mean(freq[1]),sigma=s)
#                 if np.sum(blur) != 0:
#                     blur_copy = blur.copy()
#                     new_data = wiener_filter(subtracted,blur_copy,k=k)
                    
#                     min_splice = min(len(freq[1]),len(new_data))
#                     plt.figure()
#                     plt.plot(freq[1][:min_splice]+ offset, new_data[:min_splice])
#                     path = r'Abs_Laser/Images/One/'
#                     name = 'k='+str(k)+'_sig='+str(s)+".png"
#                     plt.title(name)
#                     plt.savefig(fname = path+name, dpi = 200)
#                 bar()
            
plt.figure()
for i in range(len(freq)):
    plt.plot(freq_shifted[i],c1_grouped[i])
    plt.plot(freq_shifted[i],c1_b_grouped[i],label=i)

# c1b_data = c1_b_grouped[0]
# sav_c1b = savgol_filter(c1b_data,window_length=1901,polyorder=2)
# max_ind = argrelextrema(sav_c1b,np.greater)
# peak_yy = sav_c1b[max_ind[0]]
# peak_xx = np.array(freq_shifted[0][max_ind[0]])
# peak_xx = np.array(freq_shifted[0][max_ind[0]])
# peak_xx = peak_xx[peak_yy < -0.05]
# peak_yy = peak_yy[peak_yy < -0.05]
# peak_y = peak_y[peak_x > 10]
# peak_x = peak_x[peak_x > 10]


# c1b_data2 = c1_b_grouped[2]
# sav_c1b2 = savgol_filter(c1b_data2,window_length=2701,polyorder=2)
# max_ind2 = argrelextrema(sav_c1b2,np.greater)
# peak_yy2 = sav_c1b2[max_ind2[0]]
# peak_xx2 = np.array(freq_shifted[2][max_ind2[0]])
# peak_xx2 = np.array(freq_shifted[2][max_ind2[0]])
# peak_xx2 = peak_xx2[peak_yy2 < -0.01]
# peak_yy2 = peak_yy2[peak_yy2 < -0.01]


# # print("Peak val diff 1: ",(peak_xx[-1]-peak_xx[1])/1e9)
# # print("Peak val diff 2: ",(peak_xx2[0]-peak_xx[0])/1e9)
# plt.scatter(peak_xx,peak_yy,color='red',marker='o')
# plt.scatter(peak_xx2,peak_yy2,color='red',marker='o')


plt.legend()
plt.show()




# FP = np.array(c4)
# FP_sav = savgol_filter(FP,window_length=151,polyorder=3)
# max_ind = argrelextrema(FP_sav,np.greater)
# peak_y = FP_sav[max_ind[0]]
# peak_x = np.array(x_axis[max_ind[0]][peak_y > -0.056])
# peak_y = peak_y[peak_y > -0.056]
# spacing = np.diff(peak_x)
#----------------------------