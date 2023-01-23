import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
"""
Order = Zr,Ag,Mo,Al
"""

R_0 = 10973731.6

def line(x,m,c):
    return m*x + c

Z = np.array([40,42,47])

wav_alpha = np.array([66.5,59.5,46.5]) * 10e-12
wav_beta = np.array([135-5,122-3,95-5])* 10e-12

alpha_sigma = np.array([3,3,3])*10e-12

inv_sqrt_wav_alpha = 1/np.sqrt(wav_alpha) 
inv_sqrt_wav_beta = 1/np.sqrt(wav_beta)

grad_beta_guess = (inv_sqrt_wav_beta[1]-inv_sqrt_wav_beta[0])/(Z[1]-Z[0])
beta_guess = [grad_beta_guess,inv_sqrt_wav_beta[1]-grad_beta_guess*Z[1]]

beta_params, beta_cov = spo.curve_fit(line,Z,inv_sqrt_wav_beta,beta_guess)

grad_alpha_guess = (inv_sqrt_wav_alpha[1]-inv_sqrt_wav_alpha[0])/(Z[1]-Z[0])
alpha_guess = [grad_alpha_guess,inv_sqrt_wav_alpha[1]-grad_alpha_guess*Z[1]]

alpha_params, alpha_cov = spo.curve_fit(line,Z,inv_sqrt_wav_alpha,alpha_guess,sigma=alpha_sigma)

# print(f"Alpha Params: grad {alpha_params[0]} and intercet {alpha_params[1]}")
# print(f"Beta Params: grad {beta_params[0]} and intercet {beta_params[1]}")

print(f"Rydberg 1: {alpha_params[0]**2} Percent differnce {100*(alpha_params[0]**2-R_0)/R_0}")
print(f"Rydberg 2: {beta_params[0]**2} Percent differnce {100*(beta_params[0]**2-R_0)/R_0}")

print(f"Screening 1: {-1*alpha_params[1]/alpha_params[0]}")
print(f"Screening 2: {-1*beta_params[1]/beta_params[0]}")

print(f"Sigma Ryd {alpha_cov[0][0]**2}")
print(f"Sigma Ryd Frac {100*alpha_cov[0][0]**2/alpha_params[0]**2}")


plt.scatter(Z,inv_sqrt_wav_alpha,color='red')
plt.scatter(Z,inv_sqrt_wav_beta,color='blue')
plt.plot(Z,line(Z,beta_params[0],beta_params[1]),color='blue')
plt.plot(Z,line(Z,alpha_params[0],alpha_params[1]),color='red')
plt.xlabel("Atomic Number (Z)")
plt.ylabel(r"$\sqrt{1 / \lambda}$")
plt.grid()
plt.show()




