
import seaborn as sns
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import scipy.optimize as spo
import scipy.signal as ssp
# import umap
# import umap.plot
#import hdbscan
import itertools
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
import scipy.optimize as spo
from alive_progress import alive_bar
# from sklearn.cluster import Birch
hep.style.use("CMS")
method_ = "BFGS"

# sns.set_theme(style="ticks")

combined_x = np.loadtxt("Compton_Effect/Imaging/combined_x.csv", delimiter=",")
combined_y = np.loadtxt("Compton_Effect/Imaging/combined_y.csv", delimiter=",")
combined_d = np.loadtxt("Compton_Effect/Imaging/combined_d.csv", delimiter=",")
combined_s = np.loadtxt("Compton_Effect/Imaging/combined_s.csv", delimiter=",")
combined_alpha = np.loadtxt("Compton_Effect/Imaging/combined_alpha.csv", delimiter=",")

plt.figure(figsize=(10,12))
d  = {"X Position (cm)":combined_x,"Y Position (cm)":combined_y}
data = pd.DataFrame(d)

g = sns.JointGrid(data=data, x="X Position (cm)", y="Y Position (cm)", marginal_ticks=False)

# Set a log scaling on the y axis
# g.ax_joint.set(yscale="log")

# Create an inset legend for the histogram colorbar
cax = g.figure.add_axes([.85, .45, .02, .2])

# Add the joint and marginal histogram plots
# g.plot_joint(
#     sns.histplot, discrete=(False, False),
#      pmax=.8, cbar=True, cbar_ax=cax, kde=False, stat='probability', log_scale=False, cumulative=False, bins=50
# )
g.plot_joint(
    sns.histplot, discrete=(False, False),
     pmax=.8, cbar=True, cbar_ax=cax, kde=False, stat='probability', log_scale=False, cumulative=False, bins=50
)
# g.plot_marginals(sns.histplot, element="step", color="#03012d")
g.plot_marginals(sns.histplot, element="step")

plt.xlabel("X Position (cm)")
plt.ylabel("Y Position (cm)")
plt.show()#cmap="light:#03012d",