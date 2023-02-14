import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
from scipy.signal import savgol_filter
import scipy.optimize as spo
import scipy as sp 
hep.style.use("CMS")

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

def chi_square(obs,exp):
    obs = np.array(obs)
    exp = np.array(exp)
    chi_val = (obs-exp)**2/exp#**2
    return sum(chi_val)

def energy_compton(x,E_0, m_e = 9.11e-31, c=3e8):
    return E_0 / (1+((E_0)/(511))*(1-x))

# For 2048 bins
# def energy_convert(x):
#     return 0.36788261 * x - 10.41037249

# For 512 bins
def energy_convert(x):
    return 1.47491391 * x - 11.41351443


def gaussian(x, a, b, c, e):
    return (a * np.exp(-((x - b) ** 2) / (2 * c ** 2)) + e)

data_cs = pd.read_csv(r"Compton_Effect\Data\Session_4_10_02_2023\80_degrees.csv",skiprows=0)
compton_initial_load = np.array(data_cs['compton'])
new_channel = batch(compton_initial_load,4)[0]
new_channel = energy_convert(new_channel)

compton = []
straight = [] 
for i in range(1,11):
    path = "Compton_Effect\Data\Session_4_10_02_2023/"+str(10*i)+"_degrees.csv"
    data = pd.read_csv(path,skiprows=0)
    compton.append(batch(data['compton'],4)[1])
    straight.append(batch(data['straight'],4)[1])

fit_means = []
energy_error = []
for i in range(0,len(compton)):
    plt.figure(i)
    # plt.figure(i)
    compton_savgol = savgol_filter(compton[i]-straight[i],window_length=91,polyorder=4)
    compton_reduced = compton[i]-straight[i]
    y_error = np.sqrt(np.array(compton[i])+np.array(straight[i]))
    # if i==3:
    #     channel_splice = new_channel[(new_channel>450) & (new_channel < 650)]
    #     y_axis_splice = compton_reduced[(new_channel>450) & (new_channel < 650)]
    #     compton_savgol_spliced = compton_savgol[(new_channel>450) & (new_channel < 650)]
    #     index_splice = np.argmax(compton_savgol_spliced)
    #     y_error_splice = y_error[(new_channel>450) & (new_channel < 650)]
    #     sigma_guess_splice = 50
    #     params_splice, cov_splice = spo.curve_fit(gaussian,channel_splice,y_axis_splice,[compton_savgol_spliced[index_splice],channel_splice[index_splice],sigma_guess_splice,0],sigma=y_error_splice)
    #     print("Mean energy splice = ",params_splice[1])




  
    index = np.argmax(compton_savgol)
    length = len(new_channel)
    sigma_guess = 10
    print("REEEE",sigma_guess)
    params, cov = spo.curve_fit(gaussian,new_channel,compton_reduced,[compton_savgol[index],new_channel[index],sigma_guess,0],sigma=y_error)
    if i == 3:
        print("Non-Splice mean ", params[1])
    fit_means.append(params[1])
    energy_error.append(params[2])
    plt.plot(new_channel,compton_reduced,alpha=0.5)
    plt.plot(new_channel,compton_savgol)
    plt.plot(new_channel,gaussian(new_channel,*params),color='black')
    # plt.show()

angle_plot = np.arange(10,100,0.1)
angle = np.array([10, 20, 30 ,40 ,50 ,60 ,70 ,80, 90, 100]) * np.pi / 180
params, cov = spo.curve_fit(energy_compton,np.cos(angle),fit_means,[622])
plt.figure(len(compton))

print(params)
chi = chi_square(fit_means,energy_compton((np.cos(angle)),E_0=params))
print("Chi Squared Value" , chi)

chi_2 = chi_square(fit_means, energy_compton(np.cos(angle),622))
print("Theoretical Chi^2 ",chi_2)

plt.plot(np.cos(angle_plot),energy_compton(np.cos(angle_plot),622),label='Theoretical')
plt.scatter(np.cos(angle),fit_means,marker='x',color='black',label='Experimental data')
plt.plot(np.cos(angle_plot),energy_compton((np.cos(angle_plot)),E_0=params),label='Fit to data')

# Using Gaussian fit errors
plt.errorbar(np.cos(angle),fit_means,np.array(energy_error),ls='None',color='black',capsize=5,label=r"$1\sigma$ error") 

# Using Compton curve-fit errors
plt.errorbar(np.cos(angle),fit_means,np.zeros_like(fit_means)+np.sqrt(cov[0][0]),ls='None',color='blue',capsize=5,label=r"$1\sigma$ error C") 

plt.xlim(-0.2,1)
plt.xlabel(r"$\cos \theta$")
plt.ylabel("Energy (keV)")
plt.grid(alpha=0.8)
plt.legend(loc="upper left")
plt.show()
