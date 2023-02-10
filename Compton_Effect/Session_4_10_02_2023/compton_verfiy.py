import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
from scipy.signal import savgol_filter
hep.style.use("CMS")

data_cs = pd.read_csv(r"Compton_Effect\Data\Session_4_10_02_2023\80_degrees.csv",skiprows=0)
channel = data_cs['n_1']

compton = []
straight = [] 
for i in range(3,9):
    path = "Compton_Effect\Data\Session_4_10_02_2023/"+str(10*i)+"_degrees.csv"
    data = pd.read_csv(path,skiprows=0)
    compton.append(data['compton'])
    straight.append(data['straight'])

# Is there a nicer way to do this?
fig, ax = plt.subplots(1,6)
for i in range(0,len(compton)):
    compton_savgol = savgol_filter(compton[i]-straight[i],window_length=501,polyorder=3)
    ax[i].plot(channel,compton[i]-straight[i])
    ax[i].plot(channel,compton_savgol)

plt.show()




# print(data_cs)
# channel = data_cs['n_1']
# compton = data_cs['compton']
# straight = data_cs['straight']
# compton_reduced = compton-straight
# compton_savgol = savgol_filter(compton_reduced,window_length=501,polyorder=3)

# fig, ax = plt.subplots(1,3)
# ax[0].plot(channel,straight)
# ax[1].plot(channel,compton)
# ax[2].plot(channel,compton_reduced)
# ax[2].plot(channel,compton_savgol)

# ax[0].set_title("Straight through radiation")
# ax[1].set_title("Compton inc. Background")
# ax[2].set_title("Compton Background Reduced")

# plt.show()