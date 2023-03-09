import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from sklearn.preprocessing import *
hep.style.use("CMS")

data = pd.read_csv(r"Abs Laser\Data\07-03-2023\Run 2\FINE2.CSV")
data_B = pd.read_csv(r"Abs Laser\Data\07-03-2023\Run 2\FINEB2.CSV")



x_axis = data['in s']
# channel_1 = np.array(data['C1 in V'])
# channel_2 = np.array(data['C2 in V'])
channel_1 = np.array(data['C1 in V'])
# channel_2 = np.array(data['C3 in V'])


channel_1_B = np.array(data_B['C1 in V'])

fig, ax = plt.subplots(1,2)
ax[0].plot(x_axis,channel_1,label="Channel 1")
ax[0].plot(x_axis,channel_1_B/max(channel_1_B)*max(channel_1),label="Channel 1B")
ax[1].plot(x_axis,channel_1-channel_1_B/max(channel_1_B)*max(channel_1),label="HF Splitting")

# plt.plot(x_axis,channel_2)
ax[0].grid(alpha=0.5)
ax[1].grid(alpha=0.5)

ax[0].legend(loc="upper right")
ax[1].legend(loc="upper right")

ax[0].set_xlabel("Time (seconds)")
ax[0].set_ylabel("Voltage")

ax[1].set_xlabel("Time (seconds)")
ax[1].set_ylabel("Voltage")
plt.show()

