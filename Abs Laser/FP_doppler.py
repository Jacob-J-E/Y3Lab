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

# data = pd.read_csv(r"Abs Laser\Data\08-03-2023\DopplerFP.CSV")
# data_DB_free = pd.read_csv(r"Abs Laser\Data\08-03-2023\DopplerFreeFP.CSV")
data = pd.read_csv(r"Abs Laser\Data\10-03-2023\ALLB2.CSV")
data_DB_free = pd.read_csv(r"Abs Laser\Data\10-03-2023\ALL2.CSV")

print(data_DB_free)

x_axis = data_DB_free['in s']
c1 = data_DB_free['C1 in V']
c2 = data_DB_free['C2 in V']
c3 = data_DB_free['C3 in V']
c4 = data_DB_free['C4 in V']


c1_B = data['C1 in V']
x2 = data['in s']
print("OG C1B len: ",len(c1_B))

c1_B = c1_B[x2 > min(x_axis)]
c1_B = c1_B[x2 < max(x_axis)]

print("x1 diff ",np.diff(x_axis)[5])
print("x2 diff ",np.diff(x2)[5])


x2 = x2[x2 > min(x_axis)]
x2 = x2[x2 < max(x_axis)]
print("X data:")
print("Mins: ",min(x_axis),min(x2))
print("Maxs: ",max(x_axis),max(x2))
print("Lengths: ",len(x_axis),len(x2))

print("C1 len: ",len(c1))
print("C1B len: ",len(c1_B))

cubic_spline = spi.CubicSpline(x_axis,c4)
new_x = np.linspace(min(x_axis),max(x_axis),1000)
new_y = cubic_spline(new_x)
fig, ax = plt.subplots(1,2)
ax[0].plot(x_axis,c1,label="Doppler Free")
ax[0].plot(x_axis,c2,label="Broad Channel")
ax[0].plot(x_axis,c3,label="Laser input?")
ax[0].plot(x_axis,c4,label="FP Channel",alpha=0.5)
ax[0].plot(new_x,new_y,label="Cubic Spline",color="black")
ax[0].plot(x_axis,savgol_filter(c4,window_length=501,polyorder=3),label="FP Savgol",color='blue',ls='--')


ax[1].plot(x_axis,c1-c1_B/max(c1_B)*max(c1),label="HF Splitting")
# ax[1].plot(x_axis,10*c4,label="FP Channel",alpha=1,color='red')

# ax[1].plot(x2,data['C1 in V'],label='C1')
# ax[1].plot(x2,data['C2 in V'],label='C2')
# ax[1].plot(x2,data['C3 in V'],label='C3')
# ax[1].plot(x2,data['C4 in V'],label='C4')

ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')

plt.xlabel("Time (seconds)")
plt.ylabel("Voltage (V)")
plt.legend(loc="best")
plt.grid(alpha=0.5)
plt.show()



