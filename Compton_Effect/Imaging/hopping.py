import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import scipy.optimize as spo
import scipy.signal as ssp
hep.style.use("CMS")

def loss_function(coordinates: list, alpha: np.array, d:np.array, s:np.array):
    X, Y = coordinates
    theta = np.arctan2(Y,(X-s))
    phi = np.arctan2(Y,(d-X))
    inner_value = np.pi - alpha - theta - phi
    return np.sum(np.abs(inner_value)**2)


# alpha_test = np.array([0.5,0.5,0.5,0.5])
# d_test = np.array([2,2,2,2])
# s_test = np.array([2,2,2,2])

alpha_test = np.array([0.6719631716,0.6719631716,0.6719631716,0.6719631716])
d_test = np.array([8,8,8,8])
s_test = np.array([1,1,1,1])

X_guess = (d_test[0]+s_test[0])/2
# X_guess = d_test[0]-s_test[0]

Y_guess = ((d_test[0]-s_test[0]))/(np.tan(alpha_test[0]))

print("X Guess ", X_guess)
print("Y Guess ", Y_guess)

result = spo.basinhopping(func=loss_function, x0=[X_guess,Y_guess], niter=500, minimizer_kwargs = {"args":(alpha_test,d_test,s_test),"method":'BFGS'})
print(result)

# X = 5 
# Y = 10
# s = 1
# d = 8
# alpha = 0.67
# theta = 1.28
# phi = 1.19


