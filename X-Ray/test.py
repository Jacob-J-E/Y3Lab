import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import mplhep as hep
hep.style.use("ATLAS")


data = pd.read_csv(r"X-Ray\Data\NaCl_Initial.csv",skiprows=0)
print(data)
data = data.rename(columns={"&b / ":"angle","R_0 / 1/s":"count"})

angle = data["angle"]
count = data["count"]

x = angle
y = count
# x = np.linspace(0,2*np.pi,100)
# y = np.sin(x) + np.random.random(100) * 0.2
yhat = savgol_filter(y, 51, 3) # window size 51, polynomial order 3

plt.plot(x,y)
plt.plot(x,yhat, color='red')
plt.grid()
plt.xlabel(r"Scattering Angle \ $\deg$")
plt.ylabel(r"Count Rate \ $s^{-1}$")
plt.show()