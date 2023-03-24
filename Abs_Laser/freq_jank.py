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
#/Abs_Laser/freq_jank.py
# C:\Users\Ellre\Desktop\Physics Degree\LAB\Year 3 Lab\Y3Lab\Abs_Laser\freq_jank.py
def Gauss(x,A,mu,sigma):
    return A * np.exp(-(x-mu)**2/(2*sigma**2))

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


#loading in data
'''
loading in data
'''
# data = pd.read_csv(r"Abs_Laser\Data\14-03-2023\DUB03B.CSV")
# data_DB_free = pd.read_csv(r"Abs_Laser\Data\14-03-2023\DUB03.CSV")

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
#------------------------------------------------------------------------------------------
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
FP = np.array(c4)
FP_sav = savgol_filter(FP,window_length=151,polyorder=3)
max_ind = argrelextrema(FP_sav,np.greater)
peak_y = FP_sav[max_ind[0]]
peak_x = np.array(x_axis[max_ind[0]][peak_y > -0.056])
peak_y = peak_y[peak_y > -0.056]
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
av_points = 3
for i in range(len(array_spacing)):
    xvals = []
    if(len(array_x_peaks[i]) >= av_points):
       print("Option A")
       separation = int(len(array_x_peaks[i])/av_points)
       xvals = np.linspace(min(array_x_peaks[i]), max(array_x_peaks[i]), separation)
       yinterp = np.interp(xvals, array_x_peaks[i], array_spacing[i])
       inter_y.append(yinterp)
       inter_x.append(xvals)
    else: 
        print("Option B")
        xvals = np.array([min(array_x_peaks[i]), max(array_x_peaks[i])])
        g = (array_spacing[i][-1] - array_spacing[i][0])/(array_x_peaks[i][-1] - array_x_peaks[i][0])
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
for i in range(len(FSR_array)):
    scaling.append(((3e8/(2*22e-2))/FSR_array[i][0]))

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
for i in range(len(freq)):
    freq_flatten.append(freq[i].tolist())

freq_flatten = list(chain.from_iterable(freq_flatten))


offset = (384.230406373e12-1.7708439228e9) - (-4.9668e9)
freq_shifted = []
for  i in range(len(freq)):
    freq_shifted.append(freq[i]+offset)
freq_shifted =  np.array(freq_shifted)
# print(len(freq_flatten))
# print(len(x_axis))
# plt.plot(x_axis,c1)
# plt.plot(x_axis,c4)
# plt.plot(x_axis,c1_B)
plt.figure()
plt.plot(x_axis,c4)
plt.scatter(peak_x,peak_y)
plt.figure()
plt.scatter(peak_x[:-1],spacing)
plt.scatter(peak_x_spliced,spacing_spliced)
plt.figure()
# for cluster in np.unique(clusters):
#     plt.scatter(points[clusters == cluster, 0], points[clusters == cluster, 1], label=f"Cluster {cluster}")
for i in range(len(array_spacing)):
    plt.scatter(array_x_peaks[i],array_spacing[i], label = f'cluster {i}' )
    plt.plot(inter_x[i], inter_y[i], '-x', label = f'cluster inter {i}')
plt.figure()
for i in range(len(grouped_x_peaks_inter)):
    plt.scatter(grouped_x_peaks_inter[i],grouped_spacing_inter[i])

for i in range(len(x_axis_grouping)):
    y_array = [0.0004 for i in range(len(x_axis_grouping[i]))]
    plt.scatter(x_axis_grouping[i],y_array)

plt.figure()
for i in range(len(freq)):
    plt.plot(freq[i],c1_grouped[i])
    plt.plot(freq[i],c1_b_grouped[i])
#plt.figure()
# plt.scatter(freq_flatten,c1)
plt.figure()
sigma = 5e6
for i in range(len(freq)):
    plt.plot(freq[i],np.array(c1_grouped[i])-np.array(c1_b_grouped[i])+0.2,alpha=0.5)
for i in range(len(freq)):
    print(i)
    subtracted = np.array(c1_grouped[i])-np.array(c1_b_grouped[i])
    blur = Gauss(freq[i],A=max(subtracted)/2,mu=np.mean(freq[i]),sigma=sigma)
    if np.sum(blur) != 0:
        blur_copy = blur.copy()
        new_data = wiener_filter(subtracted,blur_copy,k=0.008)
        min_splice = min(len(freq[i]),len(new_data))
        plt.plot(freq[i][:min_splice],new_data[:min_splice])
plt.figure()
for i in range(len(freq)):
    plt.plot(freq_shifted[i],c1_grouped[i])
    plt.plot(freq_shifted[i],c1_b_grouped[i],label=i)

c1b_data = c1_b_grouped[0]
sav_c1b = savgol_filter(c1b_data,window_length=1901,polyorder=2)
max_ind = argrelextrema(sav_c1b,np.greater)
peak_yy = sav_c1b[max_ind[0]]
peak_xx = np.array(freq_shifted[0][max_ind[0]])
peak_xx = np.array(freq_shifted[0][max_ind[0]])
peak_xx = peak_xx[peak_yy < -0.05]
peak_yy = peak_yy[peak_yy < -0.05]
# peak_y = peak_y[peak_x > 10]
# peak_x = peak_x[peak_x > 10]


c1b_data2 = c1_b_grouped[2]
sav_c1b2 = savgol_filter(c1b_data2,window_length=2701,polyorder=2)
max_ind2 = argrelextrema(sav_c1b2,np.greater)
peak_yy2 = sav_c1b2[max_ind2[0]]
peak_xx2 = np.array(freq_shifted[2][max_ind2[0]])
peak_xx2 = np.array(freq_shifted[2][max_ind2[0]])
peak_xx2 = peak_xx2[peak_yy2 < -0.01]
peak_yy2 = peak_yy2[peak_yy2 < -0.01]


# print("Peak val diff 1: ",(peak_xx[-1]-peak_xx[1])/1e9)
# print("Peak val diff 2: ",(peak_xx2[0]-peak_xx[0])/1e9)
plt.scatter(peak_xx,peak_yy,color='red',marker='o')
plt.scatter(peak_xx2,peak_yy2,color='red',marker='o')


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