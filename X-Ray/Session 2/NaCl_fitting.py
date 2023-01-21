import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import scipy.optimize as spo
hep.style.use("ATLAS")


def double_Gauss(x,A1,mu1,sigma1,A2,mu2,sigma2):
    term_1 = A1 * np.exp(-(x-mu1)**2/(2*sigma1)**2)
    term_2 = A2 * np.exp(-(x-mu2)**2/(2*sigma2)**2)
    return term_1 + term_2

def line(x,m,c):
    return m*x + c

def exp(x,A,m,c):
    return A * np.exp(-1 * m * x) + c

def line_double_gauss(x,A1,mu1,sigma1,A2,mu2,sigma2,m,c):
    return 1

# Load in data
data = pd.read_csv(r"X-Ray\Data\16-01-2023\NaCl Full Data.csv",skiprows=0)
print(data)

# Split data into differnt variables
angle = data['angle']
wav = data['wav / pm']
energy = data['E / keV']
count_0 = data['R_0 / 1/s']

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# Line Data
angle_line = np.concatenate((angle[(angle > 7.9) & (angle < 12.1)],angle[(angle > 4.4) & (angle < 6.1)]))
count_line = np.concatenate((count_0[(angle > 7.9) & (angle < 12.1)],count_0[(angle > 4.4) & (angle < 6.1)]))
print(min(angle_line),max(angle_line))

lambda_guess = len(angle_line) / sum(angle_line)

line_guess = [max(count_line),lambda_guess,max(count_line)]
line_params, line_cov = spo.curve_fit(exp,angle_line,count_line,line_guess)

# plt.plot(angle,count_0,color='blue',label="Experimental Data")
# plt.plot(angle,exp(angle,*line_params),color='red',label="Exponential Background Fit-Line")
# plt.plot(angle,count_0-exp(angle,*line_params),color='black',label="Background-Reduced Data")
# plt.grid()
# plt.show()



fig, ax = plt.subplots(figsize=[8,5])

ax.plot(angle,count_0,color='blue',label="Experimental Data")
ax.plot(angle,exp(angle,*line_params),color='red',label="Exponential Background Fit-Line")

# axins = zoomed_inset_axes(ax, 0.8)
# axins.plot(angle,count_0-exp(angle,*line_params),color='black',label="Background-Reduced Data")
# x1, x2, y1, y2 = 5, 10, min(count_0), max(count_0)
# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)

# plt.xticks(visible=False)
# plt.yticks(visible=False)

ax2 = fig.add_axes([0.6, 0.5, 0.3, 0.35])
ax2.plot(angle[(angle>5)],count_0[(angle>5)]-exp(angle[(angle>5)],*line_params),color='black',label="Background-Reduced Data")
x1, x2, y1, y2 = 5, max(angle), min(count_0), max(count_0)
ax2.set_xlim(x1, x2)
ax2.set_ylim(y1, y2)
ax2.set_title("Background-Reduced Spectrum")
ax.legend(loc="upper left")
ax.set_xlabel("Angle (degrees)")
ax.set_ylabel("Count rate /s")
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
ax.grid()
# plt.draw()
plt.show()