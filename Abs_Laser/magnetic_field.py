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
hep.style.use("CMS")



voltage = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
B_field = [7.51, 8.33, 9.15, 9.97, 10.80, 11.61, 12.43, 13.26, 14.07, 14.89, 15.70, 16.54, 17.36, 18.17, 19.00]


plt.scatter(voltage,B_field)
plt.xlabel("Voltage (V)")
plt.ylabel("Magnetic Field (mT)")
plt.show()