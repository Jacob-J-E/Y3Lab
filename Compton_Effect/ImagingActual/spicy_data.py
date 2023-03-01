import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
from scipy.signal import savgol_filter
import scipy.optimize as spo
from scipy.signal import argrelextrema
import scipy as sp
hep.style.use("CMS")
method_ = "L-BFGS-B"

def batch(data: np.array ,batches: int):
    length = len(data)
    if batches == 1:
        return np.array([i for i in range(0,int(length/batches))]),np.array(data)

    data = np.array(data)

    if length % batches != 0:
        print("Invalid batching shape")
        exit()

    new_channel = [i for i in range(0,int(length/batches))]
    new_count = []

    for i in range(0,length,batches):
        new_count.append(np.sum(data[i:i+batches]))

    return np.array(new_channel),np.array(new_count)

def chi_square(obs,exp):
    obs = np.array(obs)
    exp = np.array(exp)
    chi_val = (obs-exp)**2/exp#**2
    return sum(chi_val)

def energy_compton(x,E_0, m_e = 9.11e-31, c=3e8):
    return E_0 / (1+((E_0)/(511))*(1-x))

# For 2048 bins
def energy_convert(x):
    return  0.36788261 * x -10.41037249

def gaussian(x, a, b, c, e):
    return (a * np.exp(-((x - b) ** 2) / (2 * c ** 2)) + e)

def compton_angle(E: np.array, E_0: float, m_e = 9.11e-31, c=3e8):
    val = np.arccos(511/E_0 - 511/E + 1)
    return val

def d_theta_d_E(theta,E_0):
    num = (1+E_0/511 -E_0/511 * np.cos(theta))**2
    denom = E_0**2 / 511 * np.sin(theta)
    return num/denom

def sigma_alpha(theta,sigma_E,E_0):
    return sigma_E * d_theta_d_E(theta,E_0)


def loss_function(coordinates: list, alpha: np.array, sigma_alpha: np.array, d:np.array, s:np.array):
    alpha = np.array(alpha)
    d = np.array(d)
    s = np.array(s)
    X, Y = coordinates
    theta = np.arctan2(Y,(X-s))
    phi = np.arctan2(Y,(d-X))
    percentage_difference = 100* (sigma_alpha/alpha)
    # inner_value = (np.pi - alpha - theta - phi)*percentage_difference
    inner_value = (np.pi - alpha - theta - phi)*percentage_difference

    return np.sum(np.abs(inner_value)**2)

def loss_minimizer(alpha:np.array,sigma_alpha, d:np.array, s:np.array):
    alpha = np.array(alpha)
    s = np.array(s)
    d = np.array(d)
    X_guess = (d[0]-s[0])/2
    Y_guess = ((d[0]-s[0]))/(np.tan(alpha[0]))

    X_guess = 11.5
    Y_guess = 6

    res_x = []
    res_y = []
    x_err = []
    y_err = []

    for i in range(0,4):
        result = spo.basinhopping(func=loss_function, x0=[X_guess,Y_guess], niter=200, T=0, minimizer_kwargs = {"args":(alpha,sigma_alpha,d,s),"method":method_})

        inv_hessian = result.lowest_optimization_result.hess_inv.todense()
        # inv_hessian = result.lowest_optimization_result.hess_inv

        det_inv_hessian = inv_hessian[0][0] * inv_hessian[1][1] - inv_hessian[0][1] * inv_hessian[1][0]

        res_x.append(result.x[0])
        res_y.append(result.x[1])
        # x_err.append(np.sqrt(inv_hessian[1][1]/det_inv_hessian))
        # y_err.append(np.sqrt(inv_hessian[0][0]/det_inv_hessian))
        x_err.append(np.sqrt(inv_hessian[0][0]))
        y_err.append(np.sqrt(inv_hessian[1][1]))

    res_x =  np.array(res_x)
    res_y = np.array(res_y)
    x_err = np.mean(np.abs(np.array(x_err)))
    y_err = np.mean(np.abs(np.array(y_err)))

    return [np.mean(res_x),np.mean(res_y),3*x_err,3*y_err]


directory = os.fsencode('Compton_Effect/ImagingActual/SpicyData')

background = pd.read_csv(filepath_or_buffer=r'Compton_Effect/ImagingActual/SpicyData/background.csv',skiprows=0)

list_files = []
s_values = []
for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if 's_' in filename and filename != 's_0 d_30.csv':
            #print(filename[2:5].replace('d', ''))
            s_values.append(float(filename[2:5].replace('d', '')))
            list_files.append(filename)

s_values = np.array(np.sort(s_values))
print("s vals",s_values)
d_values = np.zeros_like(s_values)+32

list_files_sorted = sorted(list_files, key= lambda x: float(x[2:5].replace('d', '')))
print(list_files_sorted)
data_sets = []
for data in list_files_sorted:
    new_data = pd.read_csv(filepath_or_buffer=r'Compton_Effect/ImagingActual/SpicyData/'+str(data),skiprows=0)
    new_data = np.array(new_data['N_1'])
    data_sets.append(new_data)


bin_numbers = np.array([i for i in range(len(data_sets[0]))])

mean_bin_list = []
standard_dev_bin_list = []
for i in range(len(data_sets)):
    print(i)
    sav_gol = savgol_filter(data_sets[i],101,3)
    index = argrelextrema(sav_gol, np.greater)
    sav_gol_max = sav_gol[index]
    bin_max = bin_numbers[index]
    sav_gol_max, bin_max =  zip(*sorted(zip(sav_gol_max, bin_max), reverse=True))
    guess = [sav_gol_max[0],bin_max[0], 10,0]
    if i == 9:
        guess = [2,1090, 10,0]

    params, cov = spo.curve_fit(gaussian,bin_numbers,data_sets[i],guess, bounds= ((0,0,0,0),(np.inf,np.inf,np.inf,np.inf)))
    plt.plot(bin_numbers,data_sets[i])
    plt.plot(bin_numbers,savgol_filter(data_sets[i],21,3))
    plt.plot(bin_numbers,gaussian(bin_numbers,*params))
    mean_bin_list.append(params[1])
    standard_dev_bin_list.append(params[2])
    print(f'Mean Bin Value {params[1]:.4g} +/- {params[2]:.4g}')
    plt.show()







    