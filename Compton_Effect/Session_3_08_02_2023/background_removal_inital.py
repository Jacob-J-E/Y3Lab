import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import pandas as pd
import math
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
import xraydb
import mplhep as hep
from matplotlib import pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

hep.style.use("ATLAS")

#Gain used:- 3 

data_cs = pd.read_csv(r"Compton_Effect\Data\Session_3_08_02_2023\Background and Cs137.csv",skiprows=0)
data_co = pd.read_csv(r"Compton_Effect\Data\Session_3_08_02_2023\Background and Co57.csv",skiprows=0)
data_am = pd.read_csv(r"Compton_Effect\Data\Session_3_08_02_2023\Background and am241.csv",skiprows=0)
print(data_cs.columns)
print(data_co.columns)
print(data_am.columns)


data_cs_n = [i for i in range(0,len(data_cs['E_1 / keV']))]
data_cs_source = data_cs['Cs137'].to_numpy()
data_cs_background = data_cs['Background'].to_numpy()

data_co_n = data_co['n_1'].to_numpy()
data_co_source = data_co['Co_57'].to_numpy()
data_co_Background = data_co['Background'].to_numpy()

data_am_n = data_am['n_1'].to_numpy()
data_am_source = data_am['background'].to_numpy()
data_am_Background = data_am['Am241'].to_numpy()


plt.figure(1)
plt.plot(data_am_n,data_cs_source, label = 'Cs-137')
plt.plot(data_am_n,data_cs_background, label = 'background')
plt.plot(data_am_n,data_cs_source-data_cs_background, label = 'background subtracted')
plt.legend()

plt.figure(2)
plt.plot(data_co_n,data_co_source, label = 'Co-57')
plt.plot(data_co_n,data_co_Background, label = 'background')
plt.plot(data_co_n,data_co_source-data_co_Background, label = 'background subtracted')
plt.legend()

plt.figure(3)
plt.plot(data_am_n,data_am_source, label = 'Am-241')
plt.plot(data_am_n,data_am_Background, label = 'background')
plt.plot(data_am_n,data_am_source-data_am_Background, label = 'background subtracted')
plt.legend()

plt.show()
