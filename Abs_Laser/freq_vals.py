import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
import scipy.interpolate as spi
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from sklearn.preprocessing import *
hep.style.use("CMS")



theory_freq = [384613608.6,384616653.6,384615267.6,384615300.6,384615364.6,384615486.6,384611109.6,384617949.6,384615078.6,384615150.6,384615312.6,384615579.6]

exp_freq = [384618176,384618065.1,384618011.9,384617932.3,384617795.9,384616789.7,384616733.7,384616701.5,384616639.1,384613666,384613632,384613592.2,384613556.7,384610981.1,384610940.7,384610765.5,384610653.5]


plt.scatter(theory_freq,np.zeros_like(theory_freq),color='blue',label="Theory")
plt.scatter(exp_freq,np.ones_like(exp_freq),color='orange',label="Exp")
plt.legend(loc="upper right")
plt.ylim(-1,6)
plt.show()