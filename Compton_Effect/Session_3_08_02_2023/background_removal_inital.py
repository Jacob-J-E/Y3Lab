import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import pandas as pd
import math
from scipy.signal import argrelextrema
import xraydb
import mplhep as hep
hep.style.use("ATLAS")


data_low_res = pd.read_csv(r"X-Ray\Data\24-01-2023\Plates Mo Source.csv",skiprows=0)