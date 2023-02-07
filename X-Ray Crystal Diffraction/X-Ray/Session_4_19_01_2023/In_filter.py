import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.signal import savgol_filter

hep.style.use("ATLAS")

data_NaCl = pd.read_csv(r"X-Ray\Data\20-01-2023\NaCl Baseline.csv",skiprows=0)
data_In = pd.read_csv(r"X-Ray\Data\23-01-2023\In Full Data.csv",skiprows=0)

# These variables are the same for all elements.
angle = data_NaCl['angle']
wav = data_NaCl['n&l / pm']
energy = data_NaCl['E / keV']

T_In =  data_In['T_5 / %']

plt.plot(wav,T_In)
plt.show()
