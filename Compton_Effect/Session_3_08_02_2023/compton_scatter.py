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


data_cs = pd.read_csv(r"Compton_Effect\Data\Session_3_08_02_2023\60_deg_compton_scatt_cs.csv",skiprows=0)

print(data_cs.columns)

n = data_cs['n_1']
compton = data_cs['Scatter']
background = data_cs['Background']
Difference = data_cs['Difference']


plt.plot(n,Difference)
plt.plot(n,savgol_filter(Difference,121,3))
plt.show()


