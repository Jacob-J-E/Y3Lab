import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

q = 1.6e-19

E = np.arange(10*1000*q,60*1000*q,q/10)
def bremsstrahlung_cross_section(E,E_e, Z):
    alpha = 1/137
    dsigma_dE = (4*np.pi*alpha**2*Z**2)/E * ((E+E_e)/E_e) * ((E/E_e) + (E_e/E) - 1 + np.log(E_e/E))
    return dsigma_dE

# B = bremsstrahlung_cross_section(E,max(E),11)
# plt.plot(E,B)
# plt.show()





data = pd.read_csv(r"X-Ray\Data\16-01-2023\NaCl Full Data.csv",skiprows=0)
print(data)


angle = data['angle']
wav = data['wav / pm']
energy = np.sort(data['E / keV'])
count_0 = data['R_0 / 1/s']


E_B = bremsstrahlung_cross_section(energy,max(energy),11)
plt.plot(energy,E_B*max(count_0))
plt.plot(energy,count_0)
plt.show()






