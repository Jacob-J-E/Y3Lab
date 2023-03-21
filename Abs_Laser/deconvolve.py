import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
import scipy.interpolate as spi
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
from scipy.signal import deconvolve
from scipy.signal import convolve
from sklearn.preprocessing import *

def Gauss(x,A,mu,sigma):
    return A * np.exp(-(x-mu)**2/(2*sigma**2))

x_axis = np.arange(-100,100,1)

# data = np.sin(x_axis)
data = Gauss(x_axis,3,20,1)
blur = Gauss(x_axis,5,20,5)
import scipy

plt.plot(x_axis, data,label="data")
plt.plot(x_axis, blur,label="Blur")
# con = convolve(data,np.meshgrid(blur,blur),mode='valid')
con = scipy.ndimage.convolve1d(data,blur)
con = np.array(con)
# con = np.convolve(data,blur,mode='same')
plt.plot(x_axis,con/max(con),label="Con")

plt.legend(loc='upper right')
plt.show()

