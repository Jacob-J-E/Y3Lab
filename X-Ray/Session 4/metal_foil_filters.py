import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("ATLAS")

data_ag = pd.read_csv(r"X-Ray\Data\17-01-2022\Ag_Filter_NaCl.csv",skiprows=0)
data_al = pd.read_csv(r"X-Ray\Data\17-01-2022\Al_Filter_NaCl.csv",skiprows=0)
data_mo = pd.read_csv(r"X-Ray\Data\17-01-2022\Mo_Filter_NaCl.csv",skiprows=0)
data_zr = pd.read_csv(r"X-Ray\Data\17-01-2022\Zr_Filter_NaCl.csv",skiprows=0)



angle_ag = data_ag['angle']
wav_ag = data_ag['wav / pm']
energy_ag = data_ag['E / keV']
count_0_ag = data_ag['R_0 / 1/s']
