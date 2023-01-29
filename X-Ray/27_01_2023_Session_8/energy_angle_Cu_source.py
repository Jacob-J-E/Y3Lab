import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
hep.style.use("CMS")

data = pd.read_csv(r"X-Ray\Data\28-01-2023\Cu No Filter Energy-Angle run.csv",skiprows=0)

angle = data['Angle']
counts = data['Counts']

plt.plot(angle,counts,label="Experimental Data")
plt.xlabel("Angle (Degrees)")
plt.ylabel(r"Counts $(s^{-1})$")
plt.show()