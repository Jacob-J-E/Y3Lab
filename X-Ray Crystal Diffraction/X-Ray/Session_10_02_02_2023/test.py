import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import pandas as pd
import math
from scipy.signal import argrelextrema
import xraydb

atomic_numbers_l = [74, 82, 79, 72, 73,75,76,77,78,80,81,83,84]
l_elements_high_res = ['W', 'Pb', 'Au', 'Hf', 'Ta', 'Re', 'Os', 'Ir', 'Pt', 'Hg', 'Tl','Bi', 'Po']

l_alpha_exp = np.array([11.357273788806593, 14.856025841377411, 13.453359107827593,0,0,0,0,0,0,0,0,0,0])*1e3
l_beta_exp = np.array([9.792511388784876, 12.588079827993488, 11.485332462914121,0,0,0,0,0,0,0,0,0,0])*1e3
l_gamma_exp = np.array([8.579884420912107, 10.59672656896353, 9.77571560579842,0,0,0,0,0,0,0,0,0,0])*1e3


l_alpha = []
l_beta = []
l_gamma = []

for i,col_name in enumerate(atomic_numbers_l):
    la1 = 0
    lb1 = 0
    lg1 = 0
    for name, line in xraydb.xray_lines(l_elements_high_res[i]).items():
        if name == 'La1':
            la1 = line.energy
        elif name == 'Lb1':
            lb1 = line.energy
        elif name == 'Lg1':
            lg1 = line.energy

    mu_guess_e1 = la1
    mu_guess_e2 = lb1
    mu_guess_e3 = lg1

    l_alpha.append(mu_guess_e1)
    l_beta.append(mu_guess_e2)
    l_gamma.append(mu_guess_e3)
    # print(l_elements_high_res[i], col_name)
    # print(la1)
    # print(lb1)
    # print(lg1)
    # print('------------------')


d = {'Atomic Number': atomic_numbers_l, 'l alpha theoretical (eV)': l_alpha, 'l beta theoretical (eV)': l_beta,'l gamma theoretical (eV)': l_gamma,
        'l alpha exp (eV)': l_alpha_exp, 'l beta exp (eV)': l_beta_exp, 'l gamma exp (eV)': l_gamma_exp}

data_frame = pd.DataFrame(data=d)
data_frame.to_csv(r'X-Ray\Session_10_02_02_2023\data.csv', index=False)