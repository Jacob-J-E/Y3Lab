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

res_x = []
res_y = []

# X_guess = 
# Y_guess = 10

x_err = []
y_err = []

# 
# Y_guess -= 7

for i in range(0,4):
    result = spo.basinhopping(func=loss_function, x0=[X_guess,Y_guess], niter=800, T=0, minimizer_kwargs = {"args":(alpha_test,d_test,s_test),"bounds":([min(s_test),max(d_test)],[5, 11])})
    inv_hessian = result.lowest_optimization_result.hess_inv.todense()
    print(inv_hessian)
    det_inv_hessian = inv_hessian[0][0] * inv_hessian[1][1] - inv_hessian[0][1] * inv_hessian[1][0]
    res_x.append(result.x[0])
    res_y.append(result.x[1])
    x_err.append(np.sqrt(inv_hessian[1][1]/det_inv_hessian))
    y_err.append(np.sqrt(inv_hessian[0][0]/det_inv_hessian))

res_x =  np.array(res_x)
res_y = np.array(res_y)
print("",np.mean(res_x)," +/- ",np.mean(x_err))
print("",np.mean(res_y)," +/- ",np.mean(y_err))


res_x = res_x[res_x > 0]
res_y = res_y[res_y > 0]


# X_high = np.percentile(res_x,75)
# X_low = np.percentile(res_x,25)
# Y_high = np.percentile(res_y,75)
# Y_low = np.percentile(res_y,25)

# X_cut = 1.5*(X_high - X_low)
# Y_cut = 1.5*(Y_high - Y_low)

# X_lower, X_upper = X_low - X_cut, X_high + X_cut
# X_bounded = res_x[(res_x > X_lower) & (res_x < X_upper)]

# Y_lower, Y_upper = Y_low - Y_cut, Y_high + Y_cut
# Y_bounded = res_y[(res_y > Y_lower) & (res_y < Y_upper)]

# print("X bounded mean", np.mean(X_bounded))
# print("Y bouned mean", np.mean(Y_bounded))

# X = 5 
# Y = 10
# s = 1
# d = 8
# alpha = 0.67
# theta = 1.28
# phi = 1.19


