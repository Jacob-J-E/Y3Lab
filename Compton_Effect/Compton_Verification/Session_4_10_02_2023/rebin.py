import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
import scipy as sp

data_cs = pd.read_csv(r"Compton_Effect\Data\Session_4_10_02_2023\80_degrees.csv",skiprows=0)
channel = np.array(data_cs['n_1'])
compton = np.array(data_cs['compton'])


# test_data = [1,2,3,4,5,6,7,8]
# batches = 4
# count = test_data

# length = len(test_data)

def batch(data: np.array ,batches: int):

    data = np.array(data)

    length = len(data)
    if length % batches != 0:
        print("Invalid batching shape")
        exit()

    new_channel = [i for i in range(0,int(length/batches))]
    new_count = []
    
    for i in range(0,length,batches):
        new_count.append(np.sum(data[i:i+batches]))

    return np.array(new_channel),np.array(new_count)



new_channel, new_counts = batch(compton,4)
plt.figure(1)
plt.plot(channel,compton)
plt.figure(2)
plt.plot(new_channel,new_counts)
plt.show()






