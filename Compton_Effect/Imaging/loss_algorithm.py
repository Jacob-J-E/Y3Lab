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

# result = spo.minimize(fun=loss_function, x0=[3,12], args=(alpha_test,d_test,s_test),tol=1e-5)
X_calc = []
Y_calc = []
for i in range(0,50):
    for j in range(0,50):
        result = spo.minimize(fun=loss_function, x0=[0.5*i,0.5*j], args=(alpha_test,d_test,s_test),tol=1e-5,method='BFGS')
        val = result.x
        X_calc.append(val[0])
        Y_calc.append(val[1])

X_calc = np.array(X_calc)
Y_calc = np.array(Y_calc)

X_calc = X_calc[X_calc > 0]
Y_calc = Y_calc[Y_calc > 0]



print("X mean", np.mean(X_calc))
print("Y mean", np.mean(Y_calc))



X_high = np.percentile(X_calc,75)
X_low = np.percentile(X_calc,25)
Y_high = np.percentile(Y_calc,75)
Y_low = np.percentile(Y_calc,25)



X_cut = 1.5*(X_high - X_low)
Y_cut = 1.5*(Y_high - Y_low)



X_lower, X_upper = X_low - X_cut, X_high + X_cut
X_bounded = X_calc[(X_calc > X_lower) & (X_calc < X_upper)]

Y_lower, Y_upper = Y_low - Y_cut, Y_high + Y_cut
Y_bounded = Y_calc[(Y_calc > Y_lower) & (Y_calc < Y_upper)]

print("X bounded mean", np.mean(X_bounded))
print("Y bouned mean", np.mean(Y_bounded))


# X = 5 
# Y = 10
# s = 1
# d = 8
# alpha = 0.67
# theta = 1.28
# phi = 1.19


