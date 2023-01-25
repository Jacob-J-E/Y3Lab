
import numpy as np
# Generate a sample array
a = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1,6,1])

import peakutils

# Find local maxima using the indexes function
local_maxima = peakutils.indexes(a)
print(local_maxima)