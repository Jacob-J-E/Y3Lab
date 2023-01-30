import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load in data.
data = pd.read_csv(r"X-Ray\Data\28-01-2023\2D Data NaCl Powder Cu Source.csv",skiprows=0,header=None)

# Remove NAN values and transpose data so angle is on the x-axis.
data.replace("#NAME?",0,inplace=True)
data = np.array(data,dtype=float)
data = np.rot90(data)

# Plot data.
plt.imshow(data)
plt.xlabel("Angle (Degrees)")
plt.ylabel("Energy Channel (Arb.)")
plt.show()
