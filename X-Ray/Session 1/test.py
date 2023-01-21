import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import mplhep as hep
hep.style.use("ATLAS")

#Import data
data = pd.read_csv(r"X-Ray\Data\NaCl_Initial.csv",skiprows=0)

# Rename columns for ease of yse
data = data.rename(columns={"&b / ":"angle","R_0 / 1/s":"count"})

angle = data["angle"]
count = data["count"]
x = angle
y = count

# Apply Savgol Filter
yhat = savgol_filter(y, 51, 3) # window size 51, polynomial order 3


# Plot experimental data with filter overlayed
plt.plot(x,y,label="Experimental Data")
plt.plot(x,yhat, color='red',label="Savgol Filter")

plt.xlabel(r"Scattering Angle \ $\deg$")
plt.ylabel(r"Count Rate \ $s^{-1}$")
plt.legend(loc="upper right")
plt.grid()
plt.show()