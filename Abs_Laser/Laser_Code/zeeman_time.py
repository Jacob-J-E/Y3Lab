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

global_data_search = pd.read_csv(r'Abs_Laser/Data/ZeemanCSV.csv')
files = global_data_search['Data File'].tolist()
files.remove('ZBB1')
files.remove('Z1')
directory = r"Abs_Laser\Data\21-03-2023"  # replace with the path to your directory

# upper = int(len(files[:50]))
# lower = 16
# iterations = int((upper - lower)/2)
# with alive_bar(iterations) as bar:
#     for i in range(lower,upper,2):
#         bar()
#         doppler_file = ''
#         doppler_free_file = ''
#         if 'B' in files[i]:
#             doppler_file = files[i]
#             doppler_free_file = files[i+1]
#         else:
#             doppler_file = files[i+1]
#             doppler_free_file = files[i]

#         if doppler_free_file =='Z7':
#             doppler_file = 'ZB6'


#         data = pd.read_csv(r"Abs_Laser\Data\21-03-2023\\"+str(doppler_file)+".CSV")
#         data1 = pd.read_csv(r"Abs_Laser\Data\21-03-2023\\"+str(doppler_free_file)+".CSV")

#         x_axis = data['in s']
#         c1 = data['C1 in V']
#         c1_B = data1['C1 in V']

#         c1 = np.array(c1)
#         c1_B = np.array(c1_B)

#         c1 = c1 - min(c1)
#         c1 = c1/max(c1)

#         c1_B = c1_B - min(c1_B)
#         c1_B= (c1_B/max(c1_B))

#         length_of_xaxis = len(x_axis)
#         normalized_x_axis = []
#         for j in range(len(x_axis)):
#             normalized_x_axis.append((j-(length_of_xaxis/2))/(length_of_xaxis/2))
#         normalized_x_axis = np.array(normalized_x_axis)
#         # plt.figure()
#         # plt.plot(normalized_x_axis,c1_B)
#         # plt.plot(normalized_x_axis,c1)
#         subtracted = np.array(c1_B) - np.array(c1)
#         subtracted = np.array(subtracted)
#         subtracted = subtracted - min(subtracted)
#         subtracted = subtracted/max(subtracted)
#         plt.plot(normalized_x_axis,subtracted + i, label = f'{doppler_file} and {doppler_free_file}', alpha = 1)
#         # plt.legend()
#         # plt.show()
# #plt.plot(x_axis,np.array(c1) - np.array(c1_B))
# plt.xlim(left= 0.12, right = 0.22)
# plt.legend()
# plt.show()


# global_data_search = pd.read_csv(r'Abs_Laser/Data/zeeman_list_update.csv')
# files = global_data_search['File_name'].tolist()
# directory = r"Abs_Laser\Data\24-03-2023\\"  # replace with the path to your directory

# upper = int(len(files))
# lower = 40
# iterations = int((upper - lower)/2)
# with alive_bar(iterations) as bar:
#     for i in range(lower,upper,2):
#         bar()
#         doppler_file = ''
#         doppler_free_file = ''
#         if 'B' in files[i]:
#             doppler_file = files[i]
#             doppler_free_file = files[i+1]
#         else:
#             doppler_file = files[i+1]
#             doppler_free_file = files[i]

#         if doppler_free_file =='Z7':
#             doppler_file = 'ZB6'


#         data = pd.read_csv(directory+str(doppler_file)+".CSV")
#         data1 = pd.read_csv(directory+str(doppler_free_file)+".CSV")

#         x_axis = data['in s']
#         c1 = data['C1 in V']
#         c1_B = data1['C1 in V']

#         c1 = np.array(c1)
#         c1_B = np.array(c1_B)

#         c1 = c1 - min(c1)
#         c1 = c1/max(c1)

#         c1_B = c1_B - min(c1_B)
#         c1_B= (c1_B/max(c1_B))

#         length_of_xaxis = len(x_axis)
#         normalized_x_axis = []
#         for j in range(len(x_axis)):
#             normalized_x_axis.append((j-(length_of_xaxis/2))/(length_of_xaxis/2))
#         normalized_x_axis = np.array(normalized_x_axis)
#         plt.figure()
#         plt.plot(normalized_x_axis,c1_B)
#         plt.plot(normalized_x_axis,c1)
#         subtracted = np.array(c1_B) - np.array(c1)
#         subtracted = np.array(subtracted)
#         subtracted = subtracted - min(subtracted)
#         subtracted = subtracted/max(subtracted)
#         plt.plot(normalized_x_axis,subtracted, label = f'{doppler_file} and {doppler_free_file}', alpha = 1)
#         plt.legend()
#         plt.show()
# #plt.plot(x_axis,np.array(c1) - np.array(c1_B))
# #plt.xlim(left= 0.2, right = 0.14)
# plt.legend()
# plt.show()


doppler_file = ''
doppler_free_file = ''
i = 16
if 'B' in files[i]:
    doppler_file = files[i]
    doppler_free_file = files[i+1]
else:
    doppler_file = files[i+1]
    doppler_free_file = files[i]

if doppler_free_file =='Z7':
    doppler_file = 'ZB6'


data = pd.read_csv(r"Abs_Laser\Data\21-03-2023\\"+str(doppler_file)+".CSV")
data1 = pd.read_csv(r"Abs_Laser\Data\21-03-2023\\"+str(doppler_free_file)+".CSV")

x_axis = data['in s']
c1 = data['C1 in V']
c1_B = data1['C1 in V']

c1 = np.array(c1)
c1_B = np.array(c1_B)

c1 = c1 - min(c1)
c1 = c1/max(c1)

c1_B = c1_B - min(c1_B)
c1_B= (c1_B/max(c1_B))

length_of_xaxis = len(x_axis)
normalized_x_axis = []
for j in range(len(x_axis)):
    normalized_x_axis.append((j-(length_of_xaxis/2))/(length_of_xaxis/2))
normalized_x_axis = np.array(normalized_x_axis)
# plt.figure()
# plt.plot(normalized_x_axis,c1_B)
# plt.plot(normalized_x_axis,c1)
subtracted = np.array(c1_B) - np.array(c1)
subtracted = np.array(subtracted)
subtracted = subtracted - min(subtracted)
subtracted = subtracted/max(subtracted)
plt.plot(normalized_x_axis,subtracted, label = f'{doppler_file} and {doppler_free_file}', alpha = 1)

doppler_file = ''
doppler_free_file = ''
i = 130
if 'B' in files[i]:
    doppler_file = files[i]
    doppler_free_file = files[i+1]
else:
    doppler_file = files[i+1]
    doppler_free_file = files[i]

if doppler_free_file =='Z7':
    doppler_file = 'ZB6'


data = pd.read_csv(r"Abs_Laser\Data\21-03-2023\\"+str(doppler_file)+".CSV")
data1 = pd.read_csv(r"Abs_Laser\Data\21-03-2023\\"+str(doppler_free_file)+".CSV")

x_axis = data['in s']
c1 = data['C1 in V']
c1_B = data1['C1 in V']

c1 = np.array(c1)
c1_B = np.array(c1_B)

c1 = c1 - min(c1)
c1 = c1/max(c1)

c1_B = c1_B - min(c1_B)
c1_B= (c1_B/max(c1_B))

length_of_xaxis = len(x_axis)
normalized_x_axis = []
for j in range(len(x_axis)):
    normalized_x_axis.append((j-(length_of_xaxis/2))/(length_of_xaxis/2))
normalized_x_axis = np.array(normalized_x_axis)
# plt.figure()
plt.plot(normalized_x_axis,c1_B)
plt.plot(normalized_x_axis,c1)
subtracted = np.array(c1_B) - np.array(c1)
subtracted = np.array(subtracted)
subtracted = subtracted - min(subtracted)
subtracted = subtracted/max(subtracted)
plt.plot(normalized_x_axis,subtracted, label = f'{doppler_file} and {doppler_free_file}', alpha = 1)

plt.xlim(left= 0.12, right = 0.22)
plt.legend()
plt.show()