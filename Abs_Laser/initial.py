import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from sklearn.preprocessing import *

hep.style.use("CMS")

data = pd.read_csv(r"Abs Laser\Data\06-03-2023\A\ALL2.CSV")


x_axis = data['in s']
# channel_1 = np.array(data['C1 in V'])
# channel_2 = np.array(data['C2 in V'])
channel_1 = np.array(data['C1 in V'])
channel_2 = np.array(data['C2 in V'])
# print(len(channel_2))
rel_size=0.07
window_length = round(rel_size*len(channel_2)/2)*2+1
print(window_length)

c1_filter = savgol_filter(channel_1,window_length,3)
c2_filter = savgol_filter(channel_2,window_length,3)

index_c1 = np.array(find_peaks(-1*c1_filter,distance=20))
index_c2 = np.array(find_peaks(-1*c2_filter,distance=20))
# print(index_c1)
peaks_c1 = c1_filter[index_c1[0]]
peaks_c2 = c2_filter[index_c2[0]]



c1_max = min(channel_1)
c2_max = min(channel_2)

# channel_1 = channel_1/min(channel_1)
# channel_2 = channel_2/min(channel_2)

# scaler = MinMaxScaler()
# scaler.fit(channel_1)

# c1_norm = scaler.transform(channel_1)
# c2_norm = scaler.transform(channel_2)
train_df = pd.DataFrame({'colA': channel_1})
test_df = pd.DataFrame({'colA': channel_2})

# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
# scaler = Normalizer()
scaler = StandardScaler()
scaler.fit(train_df)

c1_norm = scaler.transform(train_df)
c2_norm = scaler.transform(test_df)
# print(c1_max,c2_max)
# plt.plot(x_axis,np.log(channel_1),label="Log Channel 1")

# plt.plot(x_axis,channel_1,label="Channel 1")
# plt.plot(x_axis,channel_2,label="Channel 2")
plt.plot(x_axis,c1_norm,label="Channel 1")
plt.plot(x_axis,c2_norm,label="Channel 2")
plt.plot(x_axis,channel_2-channel_1,label="Channel Difference")
# plt.plot(x_axis,c2_norm-c1_norm,label="Channel Difference")

# # plt.plot(x_axis,c1_filter,label="Filter C1")
# # plt.plot(x_axis,c2_filter,label="Filter C2")
# # plt.scatter(x_axis[index_c1[0]],peaks_c1,color='black')
# # plt.scatter(x_axis[index_c2[0]],peaks_c2,color='red')

plt.grid(alpha=0.5)
plt.legend(loc="upper right")
plt.xlabel("Time (seconds)")
plt.ylabel("Voltage")
plt.show()

