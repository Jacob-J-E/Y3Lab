from funcs import *

data = pd.read_csv(r"Abs_Laser\Data\09-03-2023\DopplerFP.CSV")
data_DB_free = pd.read_csv(r"Abs_Laser\Data\09-03-2023\DopplerFreeFP.CSV")

# data = pd.read_csv(r"Abs_Laser\Data\17-03-2023\ZEEMAN0.CSV")
# data_DB_free = pd.read_csv(r"Abs_Laser\Data\17-03-2023\ZEEMAN0.CSV")

x_axis = data_DB_free['in s']
c1 = data_DB_free['C1 in V']           
c2 = data_DB_free['C2 in V'] 
c3 = data_DB_free['C3 in V']
c4 = data['C4 in V']
c1_B = data['C1 in V']
c2_B = data['C2 in V']


c1_B = c1_B/max(c1_B)*max(c1)


c1_B = np.array(c1_B)
c3 = np.array(c3)
c4 = np.array(c4)
c1 = np.array(c1)
x_axis = np.array(x_axis)


def normalize(x):
    # So True!
    x= x - min(x)
    return x/max(x)

c1_B = normalize(c1_B)
c3 = normalize(c3)
c4 = normalize(c4)
c1 = normalize(c1)

# peaks, _= find_peaks(c4, prominence=0.6)
# c4_peaks = c4[peaks]
# x_axis_peaks = x_axis[peaks]
# spacing = np.diff(x_axis_peaks)
# mid = (x_axis_peaks[1:] +x_axis_peaks[:-1])/2
# spacing = spacing - min(spacing)
# spacing = spacing/max(spacing)

plt.figure()
plt.plot(x_axis,c4,label="C4")
# plt.scatter(x_axis_peaks,c4_peaks)
plt.show()

plt.figure()
plt.plot(x_axis,c1,label="C1")
plt.plot(x_axis,c2,label="C2")
# plt.plot(x_axis,c3)
#plt.plot(x_axis,c4,label="C4")
plt.plot(x_axis,c1_B,label="C1B")
plt.plot(x_axis,c2_B,label="C2B")
# plt.scatter(x_axis_peaks[1:],spacing, label = 'Peak finder Spacing')
plt.legend(loc='upper right')
plt.figure()
plt.plot(x_axis,-(c1-c1_B/max(c1_B)*max(c1)))
plt.show()