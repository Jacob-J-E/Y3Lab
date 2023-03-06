import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
hep.style.use("CMS")

data = pd.read_csv(r"Abs Laser\Data\06-03-2023\A\ALL2.CSV")


x_axis = data['in s']
channel_1 = data['C1 in V']
channel_2 = data['C2 in V']

c1_max = min(channel_1)
c2_max = min(channel_2)
print(c1_max,c2_max)
plt.plot(x_axis,channel_1,label="Channel 1")
plt.plot(x_axis,channel_2,label="Channel 2")
plt.grid(alpha=0.5)
plt.legend(loc="upper right")
plt.xlabel("Time (seconds)")
plt.ylabel("Voltage")
plt.show()

