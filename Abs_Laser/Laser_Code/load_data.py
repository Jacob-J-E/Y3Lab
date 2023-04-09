from funcs import *

data = pd.read_csv(r"Abs_Laser\Data\21-03-2023\ZB0.CSV")
data_DB_free = pd.read_csv(r"Abs_Laser\Data\21-03-2023\Z0.CSV")

# data = pd.read_csv(r"Abs_Laser\Data\17-03-2023\ZEEMAN0.CSV")
# data_DB_free = pd.read_csv(r"Abs_Laser\Data\17-03-2023\ZEEMAN0.CSV")

x_axis = data_DB_free['in s']
c1 = data_DB_free['C1 in V']           
c2 = data_DB_free['C2 in V'] 
c3 = data_DB_free['C3 in V']
c4 = data_DB_free['C4 in V']
c1_B = data['C1 in V']

c1_B = c1_B/max(c1_B)*max(c1)


c1_B = np.array(c1_B)
c3 = np.array(c3)
c4 = np.array(c4*20)
c1 = np.array(c1)
x_axis = np.array(x_axis)


plt.plot(x_axis,c1)
plt.plot(x_axis,c2)
# plt.plot(x_axis,c3)
plt.plot(x_axis,c4)
plt.plot(x_axis,c1_B)

plt.show()