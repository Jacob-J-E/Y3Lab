import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.signal import savgol_filter

hep.style.use("ATLAS")

data_NaCl = pd.read_csv(r"X-Ray\Data\20-01-2023\NaCl Baseline.csv",skiprows=0)
data_Ag = pd.read_csv(r"X-Ray\Data\20-01-2023\Ag Full Data.csv",skiprows=0)
data_Al = pd.read_csv(r"X-Ray\Data\20-01-2023\Al Full Data.csv",skiprows=0)
data_Mo = pd.read_csv(r"X-Ray\Data\20-01-2023\Mo Full Data.csv",skiprows=0)
data_Zr = pd.read_csv(r"X-Ray\Data\20-01-2023\Zr Full Data.csv",skiprows=0)

# These variables are the same for all elements.
angle = data_NaCl['angle']
wav = data_NaCl['n&l / pm']
energy = data_NaCl['E / keV']


# Load in data for all four elements
count_Zr = data_Zr['R_1 / 1/s']
T_Zr =  data_Zr['T_1 / %']
