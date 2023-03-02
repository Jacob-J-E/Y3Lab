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
import seaborn as sns
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

    # X_guess = 11.5
    # Y_guess = 6

    res_x = []
    res_y = []
    x_err = []
    y_err = []

    for i in range(0,20):
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
        # print("itter: ",result.x[0]," +\- ",np.sqrt(inv_hessian[0][0]),result.x[1]," +\- ",np.sqrt(inv_hessian[1][1]))

    res_x =  np.array(res_x)
    res_y = np.array(res_y)
    print("vars",np.std(res_x),np.std(res_y),np.std(x_err),np.std(y_err))
    x_err = np.mean(np.abs(np.array(x_err)))
    y_err = np.mean(np.abs(np.array(y_err)))
    

    return [np.mean(res_x),np.mean(res_y),3*x_err,3*y_err]



directory = os.fsencode('Compton_Effect/ImagingActual/Data')

list_files = []
s_values = []
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     s_values.append(float(filename[9:12].replace('d', '')))
     list_files.append(filename)

s_values = np.array(np.sort(s_values))
print("s vals",s_values)
d_values = np.zeros_like(s_values)+32


list_files_sorted = sorted(list_files, key= lambda x: float(x[9:12].replace('d', '')))
data_sets = []
for data in list_files_sorted:
    new_data = pd.read_csv(filepath_or_buffer=r'Compton_Effect/ImagingActual/Data/'+str(data),skiprows=0)
    new_data = np.array(new_data['N_1']) - np.array(new_data['straight'])
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
    # plt.plot(bin_numbers,data_sets[i])
    # plt.plot(bin_numbers,savgol_filter(data_sets[i],21,3))
    # plt.plot(bin_numbers,gaussian(bin_numbers,*params))
    # plt.show()
    mean_bin_list.append(params[1])
    standard_dev_bin_list.append(params[2])
    #print(f'Mean Bin Value {params[1]:.4g} +/- {params[2]:.4g}')
mean_bin_list = np.array(mean_bin_list)
standard_dev_bin_list = np.array(standard_dev_bin_list)

print('----------------- GAUSSIAN MEAN BIN VALUES-------------------')
for i in range(len(mean_bin_list)):
    print(f'File: {list_files_sorted[i]}, Bin : {mean_bin_list[i]:.4g} +/- {standard_dev_bin_list[i]:.4g}')

mean_energy_list = energy_convert(mean_bin_list)
standard_dev_energy_list = energy_convert(standard_dev_bin_list)

print('----------------- GAUSSIAN MEAN ENERGY VALUES----------------')
# Print energy list
for i in range(len(mean_energy_list)):
    print(f'File: {list_files_sorted[i]}, Energy: {mean_energy_list[i]:.4g} +/- {standard_dev_energy_list[i]:.4g}')

E_0 = 661.7
# E_0 = 649.6084742
# E_0 =675
mean_angle_list =  np.pi - compton_angle(mean_energy_list,E_0)


standard_dev_angle_list = sigma_alpha(mean_angle_list,standard_dev_energy_list,E_0)
print('----------------- GAUSSIAN MEAN ANGLE VALUES-----------------')
for i in range(len(mean_angle_list)):
    print(f'File: {list_files_sorted[i]}, Angle: {mean_angle_list[i]*(180/np.pi):.4g} +/- {standard_dev_angle_list[i]:.4g} ({(100*standard_dev_angle_list[i]/mean_angle_list[i]):.4g}%)')
print('-------------------------------------------------------------')
calculated_coordinates = loss_minimizer(mean_angle_list,standard_dev_angle_list,d_values,s_values)
print(f'X: {calculated_coordinates[0]:.4g} +/- {calculated_coordinates[2]:.4g} | Y: {calculated_coordinates[1]:.4g} +/- {calculated_coordinates[3]:.4g}')


# x_range = []
# y_range = []
# for i in range(len(s_values)):
#     calculated_coordinates = loss_minimizer(mean_angle_list[i],mean_angle_list[i],d_values[i],s_values[i])
#     x_range.append(calculated_coordinates[0])
#     y_range.append(calculated_coordinates[1])


combined_x = []
combined_y = []
# print("sss",combined_s)
# print("ddd",combined_d)

for i in range(0,len(s_values)):
    # print("Iteration ",i," of ", len(combined_s))

    x_guess = float((d_values[i]+s_values[i])/2) - 12
    # y_guess = ((combined_d[i]+combined_s[i]))/(np.sin(combined_alpha[i]))
    # y_guess = float(x_guess+1)
    y_guess = np.abs(0.5*((d_values[i]-s_values[i]))/(np.tan(mean_angle_list[0])))-15

    # print("X-Guess",x_guess)
    # print("Y-Guess",y_guess)
    x_guess = 12+(-1)**np.random.randint(0,2) * np.random.randint(0,4)
    y_guess = 5+(-1)**np.random.randint(0,2) * np.random.randint(0,4)
    bounds = spo.Bounds(lb=[0,0],ub=[20,20])
    # result = spo.basinhopping(func=scatter_difference, niter=500, x0=list([x_guess,y_guess]), T=0, minimizer_kwargs = {"args":(combined_alpha[i],combined_d[i],combined_s[i]),"method":method_,"bounds":bounds})
    result = spo.basinhopping(func=loss_function, niter=40, x0=list([x_guess,y_guess]), T=0, minimizer_kwargs = {"args":(mean_angle_list[i],mean_angle_list[i],d_values[i],s_values[i]),"method":method_,"bounds":([0,20],[0,20])})

    # result = spo.basinhopping(func=scatter_difference, niter=500, x0=[x_guess,y_guess], T=0, minimizer_kwargs = {"args":(combined_alpha[i],combined_d[i],combined_s[i]),"method":'Powell',"bounds":([0,20],[0, 20])})

    if result.x[0] < 0:
        combined_x.append(0)
    else:
        combined_x.append(result.x[0])
    if result.x[1] < 0:
        combined_y.append(0)
    else:
        combined_y.append(result.x[1])

# plt.scatter(combined_x,combined_y,s=500,edgecolors='black',zorder=5,alpha=0.1)
# plt.scatter(combined_x,combined_y,s=500,zorder=5,alpha=0.8,label="Single Iteration Position")
plt.scatter(12,5,color='red',marker='x',s=300,zorder=50,label="Calculated Position")
plt.scatter(32,0,marker='x',color='black',s=300,zorder=5)
plt.scatter(2,0,marker='x',color='black',s=300,zorder=5)
plt.scatter(15,0,marker='x',color='black',s=300,zorder=5)
plt.plot([2,12],[0,5],color='black',ls='--',zorder=5,alpha=0.8)
plt.plot([15,12],[0,5],color='black',ls='--',zorder=5,alpha=0.8)
plt.plot([12,32],[5,0],color='black',ls='--',zorder=5,alpha=0.8)
plt.text(2+1.5,0+0.3,s=r"$s_{min}$")
plt.text(15+0.5,0+0.5,s=r"$s_{max}$")
plt.text(32-0.8,0+0.8,s=r"$d$")
plt.xlabel("X Position (cm)")
plt.ylabel("Y Position (cm)")
# plt.fill_between(combined_x,min(combined_y),8,color='blue',alpha=0.5)
# hist,xedge,yedge= np.histogram2d(combined_x,combined_y,bins=50)
# sns.histplot(x=combined_x,y=combined_y,kde=True,fill=True)
combined_y = np.array(combined_y)
combined_x = np.array(combined_x)
combined_x = combined_x[combined_y<8]
combined_y = combined_y[combined_y<8]
# sns.kdeplot(x=combined_x,y=combined_y,fill=True,label="Bootstrapping Probabiliy Range",levels=5,cbar=True)
sns.kdeplot(x=combined_x,y=combined_y,fill=True,levels=5,palette='pastel',cbar=True,cumulative=False,legend=True,common_norm=True,
            cbar_kws={'format':"%.2f","label":"Bootstrapping Probabiliy Ranges"})
# cbar = plt.colorbar(format="%.4f")
# sns.histplot(x=combined_x,y=combined_y,fill=True,kde=True,label="Bootstrapping Probabiliy Range",cbar=True)
#,label="Bootstrapping Probabiliy Ranges"


# plt.imshow(hist==0,
#            origin='lower',
#            cmap=plt.gray(),
#            extent=[xedge[0],xedge[-1],yedge[0],yedge[-1]])

# plt.imshow(hist, extent=[xedge[0],xedge[-1],yedge[0],yedge[-1]])
# plt.fill(combined_x,combined_y,color='blue',alpha=0.5)

plt.legend(loc='upper right')
plt.grid(alpha=0.9)
plt.xlim(0,33)
plt.ylim(-1,10)
plt.tight_layout()
plt.show()







