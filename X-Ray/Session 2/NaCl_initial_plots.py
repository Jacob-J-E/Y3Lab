import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("ATLAS")


# Load in data
data = pd.read_csv(r"X-Ray\Data\16-01-2022\NaCl Full Data.csv",skiprows=0)
print(data)

# Split data into differnt variables
angle = data['angle']
wav = data['wav / pm']
energy = data['E / keV']
count_0 = data['R_0 / 1/s']


#Initial Plots
fig, ax = plt.subplots(3)
ax[0].plot(angle,count_0)
ax[1].plot(wav,count_0)
ax[2].plot(np.sort(energy),count_0)

ax[0].set_xlabel(r"Angle (deg)")
ax[1].set_xlabel(r"Wavlength $(pm)$")
ax[2].set_xlabel(r"Energy $KeV$")

ax[0].set_ylabel(r"Count $(s^{-1})$")
ax[1].set_ylabel(r"Count $(s^{-1})$")
ax[2].set_ylabel(r"Count $(s^{-1})$")
plt.tight_layout()
plt.show()








