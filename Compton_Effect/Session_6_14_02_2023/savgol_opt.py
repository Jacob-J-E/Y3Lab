import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter



class Savgol_opt():

    def __init__(self) -> None:
        pass

    def chi_square(self,obs,exp):
        obs = np.array(obs)
        exp = np.array(exp)
        chi_val = (obs-exp)**2/exp#**2
        return sum(chi_val)