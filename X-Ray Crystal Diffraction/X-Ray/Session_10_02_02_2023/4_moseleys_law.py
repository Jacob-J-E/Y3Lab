import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import pandas as pd
import math
from scipy.signal import argrelextrema
from scipy.stats import chisquare
import xraydb
import mplhep as hep
hep.style.use("ATLAS")
plt.style.use('dark_background')

R_0 = 10973731.6
H = 6.63e-34
C = 3e8
FS = 0.00729735

def straight_line(x,m,c):
    return m * x + c

def cal_rydberg(m,factor):
    return (1/factor)*m**2

k_alpha = [8.080748431650838, 22.372552420613246, 16.044092033837604, 8.715570846388037, 4.57752299496841, 6.4190948273611665, 25.324555158862026, 17.421595970841373, 23.893491144556645, 7.681898645165174]
k_beta = [8.95544201799072, 25.105444424220387, 17.982645794347498, 9.695802325558363, 5.097266130416366, 7.072766943625728, 28.42222963004282, 19.53560367390865, 26.785539971006223, 8.403630082046032]

k_alpha = np.array(k_alpha)
k_beta = np.array(k_beta)

k_alpha_plot = np.sqrt(np.array(k_alpha)*1e3*1.6e-19 / (6.63e-34 * 3e8))
k_beta_plot = np.sqrt(np.array(k_beta)*1e3*1.6e-19 / (6.63e-34 * 3e8))

y_error_k_a = (0.05)*1e3*1.6e-19*(np.sqrt(H*C)/2)*(1/(np.sqrt(k_alpha*1e3*1.6e-19)**3))
y_error_k_b = (0.05)*1e3*1.6e-19*(np.sqrt(H*C)/2)*(1/(np.sqrt(k_beta*1e3*1.6e-19)**3))




#atomic_numbers_k_name = ['Cu', 'Ag', 'Zr', 'Zn','Ti', 'Fe', 'Sn','Mo', 'In', 'Ni']


atomic_numbers_k_name = ['Cu', 'Ag', 'Zr', 'Zn','Ti', 'Fe', 'Sn','Mo', 'In', 'Ni']
atomic_numbers_k = [29, 47, 40, 30, 22, 26, 50, 42, 49, 28]


atomic_numbers_ordered = [21,26,28,29,30,40,42,47,49,50] 

atomic_name_ordered = ['Ti','Fe','Ni','Cu','Zn','Zr','Mo','Ag','In','Sn'] 


l_alpha = [11.357273788806593, 14.856025841377411, 13.453359107827593]
l_beta = [9.792511388784876, 12.588079827993488, 11.485332462914121]
l_gamma = [8.579884420912107, 10.59672656896353, 9.77571560579842]

l_alpha = np.array(l_alpha)
l_beta = np.array(l_beta)
l_gamma = np.array(l_gamma)

l_alpha_plot = np.sqrt(np.array(l_alpha)*1e3*1.6e-19 / (6.63e-34 * 3e8))
l_beta_plot = np.sqrt(np.array(l_beta)*1e3*1.6e-19 / (6.63e-34 * 3e8))
l_gamma_plot = np.sqrt(np.array(l_gamma)*1e3*1.6e-19 / (6.63e-34 * 3e8))

y_error_l_a = 0.5*1e3*1.6e-19*(np.sqrt(H*C)/2)*(1/(np.sqrt(l_alpha*1e3*1.6e-19)**3))
print(y_error_l_a)
y_error_l_b = 0.5*1e3*1.6e-19*(np.sqrt(H*C)/2)*(1/(np.sqrt(l_beta*1e3*1.6e-19)**3))
y_error_l_g = 0.5*1e3*1.6e-19*(np.sqrt(H*C)/2)*(1/(np.sqrt(l_gamma*1e3*1.6e-19)**3))


atomic_numbers_l = [74, 82, 79]
l_elements_high_res = ['W', 'Pb', 'Au']

atomic_numbers_k_sorted, k_alpha_plot_sorted, k_beta_plot_sorted = zip(*sorted(zip(atomic_numbers_k, k_alpha_plot,k_beta_plot)))
atomic_numbers_l_sorted, l_alpha_plot_sorted, l_beta_plot_sorted,l_gamma_plot_sorted = zip(*sorted(zip(atomic_numbers_l, l_alpha_plot,l_beta_plot,l_gamma_plot)))

atomic_numbers_k_sorted = np.array(atomic_numbers_k_sorted)
atomic_numbers_l_sorted = np.array(atomic_numbers_l_sorted)
k_alpha_plot_sorted = np.array(k_alpha_plot_sorted)
k_beta_plot_sorted = np.array(k_beta_plot_sorted)
l_alpha_plot_sorted = np.array(l_alpha_plot_sorted)
l_beta_plot_sorted = np.array(l_beta_plot_sorted)
l_gamma_plot_sorted = np.array(l_gamma_plot_sorted)

k_alpha_plot_sorted_30 = k_alpha_plot_sorted[atomic_numbers_k_sorted >= 30]
k_beta_plot_sorted_30 = k_beta_plot_sorted[atomic_numbers_k_sorted >= 30]
atomic_numbers_k_sorted_30 = atomic_numbers_k_sorted[atomic_numbers_k_sorted >= 30]
y_error_k_a_30 = y_error_k_a[atomic_numbers_k_sorted >= 30]
y_error_k_b_30 = y_error_k_b[atomic_numbers_k_sorted >= 30]



l_alpha_plot_sorted_30 = l_alpha_plot_sorted[atomic_numbers_l_sorted >= 30]
l_beta_plot_sorted_30 = l_beta_plot_sorted[atomic_numbers_l_sorted >= 30]
l_gamma_plot_sorted_30 = l_gamma_plot_sorted[atomic_numbers_l_sorted >= 30]
atomic_numbers_l_sorted_30 = atomic_numbers_l_sorted[atomic_numbers_l_sorted >= 30]

k_grad_alpha_30_guess = (k_alpha_plot_sorted_30[1]-k_alpha_plot_sorted_30[0]) / (atomic_numbers_k_sorted_30[1]-atomic_numbers_k_sorted_30[0])
k_c_alpha_30_guess = k_alpha_plot_sorted_30[1] - k_grad_alpha_30_guess * atomic_numbers_k_sorted_30[1]
k_alpha_30_guess = [k_grad_alpha_30_guess,k_c_alpha_30_guess]

k_grad_beta_30_guess = (k_beta_plot_sorted_30[1]-k_beta_plot_sorted_30[0]) / (atomic_numbers_k_sorted_30[1]-atomic_numbers_k_sorted_30[0])
k_c_beta_30_guess = k_beta_plot_sorted_30[1] - k_grad_beta_30_guess * atomic_numbers_k_sorted_30[1]
k_beta_30_guess = [k_grad_beta_30_guess,k_c_beta_30_guess]

# ---- L Lines ----

l_grad_alpha_30_guess = (l_alpha_plot_sorted_30[1]-l_alpha_plot_sorted_30[0]) / (atomic_numbers_l_sorted_30[1]-atomic_numbers_l_sorted_30[0])
l_c_alpha_30_guess = l_alpha_plot_sorted_30[1] - l_grad_alpha_30_guess * atomic_numbers_l_sorted_30[1]
l_alpha_30_guess = [l_grad_alpha_30_guess,l_c_alpha_30_guess]

l_grad_beta_30_guess = (l_beta_plot_sorted_30[1]-l_beta_plot_sorted_30[0]) / (atomic_numbers_l_sorted_30[1]-atomic_numbers_l_sorted_30[0])
l_c_beta_30_guess = l_beta_plot_sorted_30[1] - l_grad_beta_30_guess * atomic_numbers_l_sorted_30[1]
l_beta_30_guess = [l_grad_beta_30_guess,l_c_beta_30_guess]

l_grad_gamma_30_guess = (l_gamma_plot_sorted_30[1]-l_gamma_plot_sorted_30[0]) / (atomic_numbers_l_sorted_30[1]-atomic_numbers_l_sorted_30[0])
l_c_gamma_30_guess = l_gamma_plot_sorted_30[1] - l_grad_gamma_30_guess * atomic_numbers_l_sorted_30[1]
l_gamma_30_guess = [l_grad_gamma_30_guess,l_c_gamma_30_guess]

abol = False

k_alpha_30_fit,k_alpha_30_cov = spo.curve_fit(straight_line,atomic_numbers_k_sorted_30,k_alpha_plot_sorted_30,k_alpha_30_guess,sigma = y_error_k_a_30,absolute_sigma = abol)
k_beta_30_fit,k_beta_30_cov = spo.curve_fit(straight_line,atomic_numbers_k_sorted_30,k_beta_plot_sorted_30,k_beta_30_guess,sigma = y_error_k_b_30,absolute_sigma = abol)
l_alpha_30_fit,l_alpha_30_cov = spo.curve_fit(straight_line,atomic_numbers_l_sorted_30,l_alpha_plot_sorted_30,l_alpha_30_guess,sigma = y_error_l_a,absolute_sigma = abol)
l_beta_30_fit,l_beta_30_cov = spo.curve_fit(straight_line,atomic_numbers_l_sorted_30,l_beta_plot_sorted_30,l_beta_30_guess,sigma = y_error_l_b,absolute_sigma = abol)
l_gamma_30_fit,l_gamma_30_cov = spo.curve_fit(straight_line,atomic_numbers_l_sorted_30,l_gamma_plot_sorted_30,l_gamma_30_guess,sigma = y_error_l_g,absolute_sigma = abol)

k_domain_invalid = np.arange(min(atomic_numbers_k_sorted),31,1)
k_domain_valid = np.arange(30,max(atomic_numbers_k_sorted)+1,1)

l_domain_invalid = np.arange(min(atomic_numbers_k_sorted),min(atomic_numbers_l_sorted)+1,1)
l_domain_valid = np.arange(min(atomic_numbers_l_sorted),max(atomic_numbers_l_sorted)+1,1)
plt.figure(figsize=(10,10))

# plt.scatter(atomic_numbers_k_sorted,k_alpha_plot_sorted, label = r'$k_{\alpha} Data$', color = u'#1f77b4',marker='x')
# plt.plot(k_domain_valid,straight_line(k_domain_valid,*k_alpha_30_fit), label = r'$k_{\alpha} fit$', color = u'#1f77b4')
# plt.plot(k_domain_invalid,straight_line(k_domain_invalid,*k_alpha_30_fit), color = u'#1f77b4', linestyle = '--')
# plt.scatter(atomic_numbers_k_sorted,k_beta_plot_sorted, label = r'$k_{\beta}$',color = u'#ff7f0e',marker='x')
# plt.plot(k_domain_valid,straight_line(k_domain_valid,*k_beta_30_fit), label = r'$k_{\beta} fit$', color = u'#ff7f0e')
# plt.plot(k_domain_invalid,straight_line(k_domain_invalid,*k_beta_30_fit), color = u'#ff7f0e', linestyle = '--')
# plt.scatter(atomic_numbers_l_sorted,l_alpha_plot_sorted, label = r'$l_{\alpha}$', color = u'#2ca02c',marker='x')
# plt.plot(l_domain_valid,straight_line(l_domain_valid,*l_alpha_30_fit), label = r'$l_{\alpha} fit$', color = u'#2ca02c')
# plt.plot(l_domain_invalid,straight_line(l_domain_invalid,*l_alpha_30_fit), color = u'#2ca02c', linestyle = '--')
# plt.scatter(atomic_numbers_l_sorted,l_beta_plot_sorted, label = r'$l_{\beta}$', color = u'#d62728',marker='x')
# plt.plot(l_domain_valid,straight_line(l_domain_valid,*l_beta_30_fit), label = r'$l_{\beta} fit$', color = u'#d62728')
# plt.plot(l_domain_invalid,straight_line(l_domain_invalid,*l_beta_30_fit), color = u'#d62728', linestyle = '--')
# plt.scatter(atomic_numbers_l_sorted,l_gamma_plot_sorted, label = r'$l_{\gamma}$', color = u'#9467bd',marker='x')
# plt.plot(l_domain_valid,straight_line(l_domain_valid,*l_gamma_30_fit), label = r'$l_{\gamma} fit$', color = u'#9467bd')
# plt.plot(l_domain_invalid,straight_line(l_domain_invalid,*l_gamma_30_fit), color = u'#9467bd', linestyle = '--')


# plt.plot(Z,moseley,label="Moseley's Law",color="#46bddf")

# # plt.plot(Z,K_alpha,label=r"$K_{\alpha} Line$",color="#e4e5e7")
# plt.plot(Z,K_beta,label=r"$K_{\beta} Line$",color="#f05464")

# plt.plot(Z,L_alpha,label=r"$L_{\alpha} Line$",color="#e87454")

# plt.plot(Z,L_beta,label=r"$L_{\beta} Line$",color="#58d474")

# plt.plot(Z,L_gamma,label=r"$L_{\gamma} Line$",color="#e8c454")
plt.figure(1,figsize=(14,10))


sigma_energy = 0.05 * 1e3 * 1.6e-19
sigma_plot = np.zeros_like(k_alpha_plot_sorted)
sigma_plot_K_alpha = sigma_energy * 0.5 * (1 /( k_alpha_plot_sorted* 6.63e-34 * 3e8))
sigma_plot_K_beta = sigma_energy * 0.5 * (1 /(k_beta_plot_sorted* 6.63e-34 * 3e8))
sigma_plot_L_alpha =  sigma_energy * 0.5 * (1 /(l_alpha_plot_sorted* 6.63e-34 * 3e8))
sigma_plot_L_beta = sigma_energy * 0.5 * (1 /(l_beta_plot_sorted* 6.63e-34 * 3e8))
sigma_plot_L_gamma =  sigma_energy * 0.5 * (1 /( l_gamma_plot_sorted*6.63e-34 * 3e8))



plt.scatter(atomic_numbers_k_sorted,k_alpha_plot_sorted, label = r'$k_{\alpha} Data$',marker='x',color='#46bddf')
plt.plot(k_domain_valid,straight_line(k_domain_valid,*k_alpha_30_fit), label = r'$k_{\alpha} fit$',color='#46bddf')
plt.plot(k_domain_invalid,straight_line(k_domain_invalid,*k_alpha_30_fit), linestyle = '--',color='#46bddf')
plt.errorbar(atomic_numbers_k_sorted,k_alpha_plot_sorted,yerr=sigma_plot_K_alpha,color='#46bddf',ls='none',marker='x',capsize=5)

plt.scatter(atomic_numbers_k_sorted,k_beta_plot_sorted, label = r'$k_{\beta}$',marker='x',color='#f05464')
plt.plot(k_domain_valid,straight_line(k_domain_valid,*k_beta_30_fit), label = r'$k_{\beta} fit$',color='#f05464')
plt.plot(k_domain_invalid,straight_line(k_domain_invalid,*k_beta_30_fit), linestyle = '--',color='#f05464')
plt.errorbar(atomic_numbers_k_sorted,k_beta_plot_sorted,yerr=sigma_plot_K_beta,color='#f05464',ls='none',marker='x',capsize=5)

 
plt.scatter(atomic_numbers_l_sorted,l_alpha_plot_sorted, label = r'$l_{\alpha}$',marker='x',color='#e87454')
plt.plot(l_domain_valid,straight_line(l_domain_valid,*l_alpha_30_fit), label = r'$l_{\alpha} fit$',color='#e87454')
plt.plot(l_domain_invalid,straight_line(l_domain_invalid,*l_alpha_30_fit), linestyle = '--',color='#e87454')
plt.errorbar(atomic_numbers_l_sorted,l_alpha_plot_sorted,yerr=sigma_plot_L_alpha,color='#e87454',ls='none',marker='x',capsize=5)

plt.scatter(atomic_numbers_l_sorted,l_beta_plot_sorted, label = r'$l_{\beta}$',marker='x',color='#58d474')
plt.plot(l_domain_valid,straight_line(l_domain_valid,*l_beta_30_fit), label = r'$l_{\beta} fit$',color='#58d474')
plt.plot(l_domain_invalid,straight_line(l_domain_invalid,*l_beta_30_fit), linestyle = '--',color='#58d474')
plt.errorbar(atomic_numbers_l_sorted,l_beta_plot_sorted,yerr=sigma_plot_L_beta,color='#58d474',ls='none',marker='x',capsize=5)

plt.scatter(atomic_numbers_l_sorted,l_gamma_plot_sorted, label = r'$l_{\gamma}$', marker='x',color='#e8c454')
plt.plot(l_domain_valid,straight_line(l_domain_valid,*l_gamma_30_fit), label = r'$l_{\gamma} fit$',color='#e8c454')
plt.plot(l_domain_invalid,straight_line(l_domain_invalid,*l_gamma_30_fit), linestyle = '--',color='#e8c454')
plt.errorbar(atomic_numbers_l_sorted,l_gamma_plot_sorted,yerr=sigma_plot_L_gamma,color='#e8c454',ls='none',marker='x',capsize=5)



atomic_numbers_ordered = [21,26,28,29,30,40,42,47,49,50] 

atomic_name_ordered = ['Ti','Fe','Ni','Cu','Zn','Zr','Mo','Ag','In','Sn'] 

atomic_numbers_ordered_l = [74, 79, 82]
atomic_name_ordered_l = ['W', 'Au', 'Pb']

for i,name in enumerate(atomic_name_ordered):
    plt.text(atomic_numbers_k_sorted[i]- 0.5, k_beta_plot_sorted[i], name, horizontalalignment='right', size='medium', color='white', weight='semibold')

plt.text(atomic_numbers_k_sorted[9]+ 1, k_beta_plot_sorted[9], r'$k_{\beta}$', horizontalalignment='left', size='large', color=u'#f05464', weight='semibold')
plt.text(atomic_numbers_k_sorted[9] + 1, k_alpha_plot_sorted[9], r'$k_{\alpha}$', horizontalalignment='left', size='large', color=u'#46bddf', weight='semibold')

for i,name in enumerate(atomic_name_ordered_l):
    plt.text(atomic_numbers_l_sorted[i]- 0.5, l_alpha_plot_sorted[i], name, horizontalalignment='right', size='medium', color='white', weight='semibold')

plt.text(atomic_numbers_l_sorted[2]+ 1, l_beta_plot_sorted[2], r'$L_{\beta}$', horizontalalignment='left', size='large', color=u'#58d474', weight='bold')
plt.text(atomic_numbers_l_sorted[2] + 1, l_alpha_plot_sorted[2], r'$L_{\alpha}$', horizontalalignment='left', size='large', color=u'#e87454', weight='bold')
plt.text(atomic_numbers_l_sorted[2] + 1, l_gamma_plot_sorted[2], r'$L_{\gamma}$', horizontalalignment='left', size='large', color=u'#e8c454', weight='bold')


k_alpha_rydberg = cal_rydberg(k_alpha_30_fit[0],3/4)
k_beta_rydberg = cal_rydberg(k_beta_30_fit[0],8/9)
l_alpha_rydberg = cal_rydberg(l_alpha_30_fit[0],5/36)
l_beta_rydberg = cal_rydberg(l_beta_30_fit[0],3/16)
l_gamma_rydberg = cal_rydberg(l_gamma_30_fit[0],21/100)


def chi_square(obs,exp):
    obs = np.array(obs)
    exp = np.array(exp)
    chi_val = (obs-exp)**2/exp**2
    return sum(chi_val)

def chi_square_reduced(obs,exp,sigma):
    obs = np.array(obs)
    exp = np.array(exp)
    chi_val = (obs-exp)**2/sigma**2
    return sum(chi_val)


# observe = [1,5,1]
# expected = [1,2,3]

# print(chi_square(observe,expected))


print(l_beta_30_fit[0])
print(l_alpha_30_fit)
print(f"Percentage Differnce k_alpha: {(100*(R_0 - k_alpha_rydberg)/R_0):.4g} with value {k_alpha_rydberg:.4g} with error, {cal_rydberg(np.sqrt(k_alpha_30_cov[0][0]),3/4):.4g}, chi - {chi_square(k_alpha_plot_sorted,straight_line(atomic_numbers_k_sorted,*k_alpha_30_fit)):.4g}")
print(f"Percentage Differnce k_beta: {(100*(R_0 - k_beta_rydberg)/R_0):.4g} with value {k_beta_rydberg:.4g} with error, {cal_rydberg(np.sqrt(k_beta_30_cov[0][0]),8/9):.4g}, chi - {chi_square(k_beta_plot_sorted,straight_line(atomic_numbers_k_sorted,*k_beta_30_fit)):.4g}")
print(f"Percentage Differnce l_alpha: {(100*(R_0 - l_alpha_rydberg)/R_0):.4g} with value {l_alpha_rydberg:.4g} with error, {cal_rydberg(np.sqrt(l_alpha_30_cov[0][0]),5/36):.4g}, chi - {chi_square(l_alpha_plot_sorted,straight_line(atomic_numbers_l_sorted,*l_alpha_30_fit)):.4g}")
print(f"Percentage Differnce l_beta: {(100*(R_0 - l_beta_rydberg)/R_0):.4g} with value {l_beta_rydberg:.4g} with error, {cal_rydberg(np.sqrt(l_beta_30_cov[0][0]),3/16):.4g}, chi - {chi_square(l_beta_plot_sorted,straight_line(atomic_numbers_l_sorted,*l_beta_30_fit)):.4g}")
print(f"Percentage Differnce l_gamma: {(100*(R_0 - l_gamma_rydberg)/R_0):.4g} with value {l_gamma_rydberg:.4g} with error, {cal_rydberg(np.sqrt(l_gamma_30_cov[0][0]),21/100):.4g}, chi - {chi_square(l_gamma_plot_sorted,straight_line(atomic_numbers_l_sorted,*l_gamma_30_fit)):.4g}")

print(f"k_alpha Q value: {k_alpha_30_fit[0]**2/R_0}. Expected Value:  {3/4}, percentage difference: {100*((3/4) - (k_alpha_30_fit[0]**2/R_0))/(3/4)} ")
print(f"k_beta Q value: {k_beta_30_fit[0]**2/R_0} with value {8/9}, percentage difference: {100*((8/9) - (k_beta_30_fit[0]**2/R_0))/(8/9)} ")
print(f"l_alpha Q value: {l_alpha_30_fit[0]**2/R_0} with value {5/36}, percentage difference: {100*((5/36) - (l_alpha_30_fit[0]**2)/R_0)/(5/36)} ")
print(f"l_beta Q value: {l_beta_30_fit[0]**2/R_0} with value {3/16}, percentage difference: {100*((3/16) - (l_beta_30_fit[0]**2)/R_0)/(3/16)} ")
print(f"l_gamma Q value: {l_gamma_30_fit[0]**2/R_0} with value {21/100}, percentage difference: {100*((21/100) - (l_gamma_30_fit[0]**2/R_0))/(21/100)} ")

print(f"k_alpha Q value: {k_alpha_30_fit[0]**2/R_0}. Expected Value:  {3/4}, percentage difference: {100*((3/4) - (k_alpha_30_fit[0]**2/R_0))/(3/4)} ")
print(f"k_beta Q value: {k_beta_30_fit[0]**2/R_0} with value {8/9}, percentage difference: {100*((8/9) - (k_beta_30_fit[0]**2/R_0))/(8/9)} ")
print(f"l_alpha Q value: {l_alpha_30_fit[0]**2/R_0} with value {5/36}, percentage difference: {100*((5/36) - (l_alpha_30_fit[0]**2)/R_0)/(5/36)} ")
print(f"l_beta Q value: {l_beta_30_fit[0]**2/R_0} with value {3/16}, percentage difference: {100*((3/16) - (l_beta_30_fit[0]**2)/R_0)/(3/16)} ")
print(f"l_gamma Q value: {l_gamma_30_fit[0]**2/R_0} with value {21/100}, percentage difference: {100*((21/100) - (l_gamma_30_fit[0]**2/R_0))/(21/100)} ")




plt.xlabel("Atomic Number",fontsize=20,labelpad=10)
plt.ylabel(r"$\sqrt{\frac{1}{\lambda}}$   $\left(m^{-1/2}\right)$",fontsize=20,labelpad=10)
plt.grid(alpha=0.5)
plt.ylim(0,1.85e5)
plt.xlim(15,90)
# plt.grid()
#plt.legend()
#plt.legend()
plt.show()


