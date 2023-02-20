import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import pandas as pd
import math
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from scipy.odr import ODR, Data, Model
import xraydb
import mplhep as hep
from matplotlib import pyplot as plt
import numpy as np

hep.style.use("ATLAS")


deg_30_data = pd.read_csv(r"Compton_Effect\Data\Session_4_10_02_2023\compton_effect\30_degrees.csv",skiprows=0)
deg_40_data = pd.read_csv(r"Compton_Effect\Data\Session_4_10_02_2023\compton_effect\40_degrees.csv",skiprows=0)
deg_50_data = pd.read_csv(r"Compton_Effect\Data\Session_4_10_02_2023\compton_effect\50_degrees.csv",skiprows=0)
deg_60_data = pd.read_csv(r"Compton_Effect\Data\Session_4_10_02_2023\compton_effect\60_degrees.csv",skiprows=0)
deg_70_data = pd.read_csv(r"Compton_Effect\Data\Session_4_10_02_2023\compton_effect\70_degrees.csv",skiprows=0)
deg_80_data = pd.read_csv(r"Compton_Effect\Data\Session_4_10_02_2023\compton_effect\80_degrees.csv",skiprows=0)
background = pd.read_csv(r"Compton_Effect\Data\Session_4_10_02_2023\compton_effect\background.csv",skiprows=0)


#--30 Degrees Fitting---------------------

bin = deg_30_data['n_1'].to_numpy()[:2047]
scatter_30 = deg_30_data['compton'].to_numpy()[:2047]
straight_30 = deg_30_data['straight'].to_numpy()[:2047]

plt.plot(bin,scatter_30)
plt.plot(bin,straight_30)
plt.plot(bin,scatter_30-straight_30)
plt.plot(bin,savgol_filter(scatter_30-straight_30,151,3))
plt.plot(bin,savgol_filter(straight_30,151,3))
plt.plot(bin,savgol_filter(scatter_30,151,3))
plt.show()

while 