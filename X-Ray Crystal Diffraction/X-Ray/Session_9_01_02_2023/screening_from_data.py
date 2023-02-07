import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xraydb

R_0 = 10973731.6
h = 6.63e-34
c = 3e8
# elements = ['H', 'He', 'Li', 'Be','B','C','N','O','F','N','Na','Mg','Al','Si','Ph','S',"Cl","Ar","K","Ca"]
def gen_sigma(Z,E):
    return Z - np.sqrt((E * 1.6e-19)/(h*c*R_0))

# ka1 = []
# kb1 = []
# for i in range(0,len(elements)):
#     for name, line in xraydb.xray_lines(elements[i], 'K').items():
#         if name == 'Ka1':
#             # ka1 = line.energy
#             ka1.append(line.energy)
#         elif name == 'Kb1':
#             kb1.append(line.energy)
# print(f"K_alpha energies: {ka1}")
# print(f"K_beta energies: {kb1}")

element_list = []
ka1 = []
kb1 = []
for i in range(1,99):
    x_ray_element = xraydb.atomic_symbol(i)
    element_list.append(str(x_ray_element))

    for name, line in xraydb.xray_lines(str(x_ray_element), 'K').items():
        x_ray_data = xraydb.xray_lines(str(x_ray_element), 'K').items()
        if name == 'Ka1':
            ka1.append(line.energy)
        elif name == 'Kb1':
            kb1.append(line.energy)

k_alpha_plot = ka1[20:50]
k_beta_plot = kb1[20:50]
Z_num = np.arange(20,50,1)


sigma_alpha = gen_sigma(np.array(Z_num),np.array(k_alpha_plot))
sigma_beta= gen_sigma(np.array(Z_num),np.array(k_beta_plot))

plt.scatter(Z_num,sigma_alpha,color='red',label=r"$K_{\alpha}$ Lines")
plt.scatter(Z_num,sigma_beta,color='blue',label=r"$K_{\beta}$ Lines")
plt.xlabel("Atomic Number (No Units)")
plt.ylabel("Screening Constant")
plt.legend(loc="upper left")
plt.show()













