
from scipy.signal import argrelextrema
import numpy as np

# Find local maxima using the argrelextrema function, returns index of local max
local_maxima = argrelextrema(ARRAY_OF_Y_VALUES, np.greater)
amplitudes = []
for x in local_maxima[0]:
    amplitudes.append(ARRAY_OF_Y_VALUES[x])

amplitudes.sort(reverse=True)