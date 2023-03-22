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
import os
hep.style.use("CMS")


global_data_search = pd.read_csv(r'Abs_Laser/Data/ZeemanCSV.csv')
files = global_data_search['Data File'].tolist()
files.remove('ZBB1')
files.remove('Z1')
directory = r"Abs_Laser\Data\21-03-2023"  # replace with the path to your directory


for i in range(len(files[20:31])-1):
    data = pd.read_csv(r"Abs_Laser\Data\21-03-2023\\"+str(files[i])+".CSV")
    data1 = pd.read_csv(r"Abs_Laser\Data\21-03-2023\\"+str(files[i+1])+".CSV")

    x_axis = data['in s']
    c1 = data['C1 in V']
    c1_B = data1['C1 in V']
    plt.plot(x_axis,(-1)**(i+1)*(np.array(c1) - np.array(c1_B))+i)
#plt.plot(x_axis,np.array(c1) - np.array(c1_B))
plt.xlim(left= 0.0145, right = 0.025)
plt.show()

































