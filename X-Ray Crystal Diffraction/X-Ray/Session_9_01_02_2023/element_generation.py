import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import pandas as pd
import math
from scipy.signal import argrelextrema
import xraydb
R_0 = 10973731.6
H = 6.63e-34
C = 3e8
FS = 0.00729735

element_list = []

for i in range(1,99):
    element_list.append(str(xraydb.atomic_symbol(i)))
print(element_list)

