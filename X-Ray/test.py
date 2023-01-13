import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import mplhep as hep
hep.style.use("ATLAS")


data = pd.read_csv(r"C:\Users\Ellre\Desktop\Physics Degree\LAB\Year 3 Lab\X Ray Diffraction\Data\NaCl Initial.csv",skiprows=1)
print(data)



# x = np.linspace(0,2*np.pi,100)
# y = np.sin(x) + np.random.random(100) * 0.2
# yhat = savgol_filter(y, 51, 3) # window size 51, polynomial order 3

# plt.plot(x,y)
# plt.scatter(x,yhat, color='red')
# plt.grid()
# plt.xlabel(r"Scattering Angle \ $\deg$")
# plt.ylabel(r"Count Rate \ $s^{-1}$")
# plt.show()