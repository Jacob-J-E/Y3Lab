import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

R_0 = 10973731.6
h = 6.63e-34
c = 3e8

alpha_energy = np.array([8.95544202, 25.10544442, 17.98264579,  9.69580233, 8.3323761, 7.07276694, 5.09726613, 19.90977414])
beta_energy = np.array([8.08074843, 22.37255242, 16.04409203, 8.71557085, 7.50820029, 6.41909483, 4.57752299, 17.75495555])
elements = ['Cu', 'Ag', 'Zr', 'Zn', 'Ni', 'Fe', 'Ti', 'Mo']
Z_Number = np.array([29, 47, 40, 30, 28, 26, 22, 42])

def gen_sigma(Z,E):
    return Z - np.sqrt((E * 1e3 * 1.6e-19)/(h*c*R_0))

sigma_assume_R_0_alpha = gen_sigma(Z_Number,alpha_energy)
sigma_assume_R_0_beta = gen_sigma(Z_Number,beta_energy)

plt.scatter(Z_Number,sigma_assume_R_0_alpha,color='red',label=r"$K_{\alpha} Lines$")
plt.scatter(Z_Number,sigma_assume_R_0_beta,color='blue',label=r"$K_{\beta} Lines$")
plt.xlabel("Atomic Number (No Units)")
plt.ylabel("Screening Constant (No Units)")
plt.legend(loc="upper right")
plt.show()
