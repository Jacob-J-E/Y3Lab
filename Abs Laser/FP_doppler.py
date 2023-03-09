import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
import scipy.interpolate as spi
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from sklearn.preprocessing import *
hep.style.use("CMS")

data = pd.read_csv(r"Abs Laser\Data\08-03-2023\DopplerFP.CSV")
data_DB_free = pd.read_csv(r"Abs Laser\Data\08-03-2023\DopplerFreeFP.CSV")

print(data_DB_free)

x_axis = data_DB_free['in s']
c1 = data_DB_free['C1 in V']
c2 = data_DB_free['C2 in V']
c3 = data_DB_free['C3 in V']
c4 = data_DB_free['C4 in V']


cubic_spline = spi.CubicSpline(x_axis,c4)
new_x = np.linspace(min(x_axis),max(x_axis),1000)
new_y = cubic_spline(new_x)
# plt.plot(x_axis,c1,label="Doppler Free")
# plt.plot(x_axis,c2,label="Broad Channel")
# plt.plot(x_axis,c3,label="FP Channel")
# plt.plot(x_axis,c4,label="Laser input?")
plt.plot(new_x,new_y,label="Cubic Spline",color="black")

plt.xlabel("Time (seconds)")
plt.ylabel("Voltage (V)")
plt.legend(loc="best")
plt.grid(alpha=0.5)
plt.show()