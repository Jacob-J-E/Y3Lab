import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import scipy.optimize as spo
import scipy.signal as ssp
hep.style.use("CMS")

def loss_function(coordinates: list, alpha: np.array, d:np.array, s:np.array):
    alpha = np.array(alpha)
    d = np.array(d)
    s = np.array(s)

    X, Y = coordinates
    theta = np.arctan2(Y,(X-s))
    phi = np.arctan2(Y,(d-X))
    inner_value = np.pi - alpha - theta - phi
    return np.sum(np.abs(inner_value)**2)

def alpha_calc(X,Y,d,s):
    theta = np.arctan2(Y,(X-s))
    phi = np.arctan2(Y,(d-X))
    alpha = np.pi - theta - phi
    return alpha

def geo_difference(theory,exp):
    diff = np.sqrt(np.sum((theory-exp)**2))
    return diff



 

X_bounds = [1,10]
Y_bounds = [1,10]
geometries = []

alpha_temp = []
s_temp = []
d_temp = []

valid_geometry = []
for x in range(X_bounds[0],X_bounds[1]+1):
    for y in range(Y_bounds[0],Y_bounds[1]+1):
        for s in range(X_bounds[0],X_bounds[1]+1):
            for d in range(X_bounds[0]+1, X_bounds[1]):
                # alpha = alpha_calc(x,y,d,s)
                # geometries.append([s,d,alpha,x,y])
                # print(x)
                if (x == 6) and (y==4):
                    valid_alpha = alpha_calc(x,y,d,s)
                    if valid_alpha >  0.11:
                        valid_geometry.append([valid_alpha,d,s])


print(valid_geometry)

valid_geometry = valid_geometry

print(valid_geometry)

# # X_guess = (valid_geometry[0][1]+valid_geometry[0][2])/2
# # X_guess = valid_geometry[0][1]-valid_geometry[0][2]
# X_guess = 8

# # Y_guess = ((valid_geometry[0][1]-valid_geometry[0][2]))/(np.tan(valid_geometry[0][0]))
# Y_guess =  3

# print("X Guess ", X_guess)
# print("Y Guess ", Y_guess)

# res_x = []
# res_y = []

# x_err = []
# y_err = []


# for i in range(0,5):
#     result = spo.basinhopping(func=loss_function, x0=[X_guess,Y_guess], niter=1000, T=0, minimizer_kwargs = {"args":(valid_geometry[0],valid_geometry[1],valid_geometry[2]),"bounds":([1,10],[5, 11])})
#     # result = spo.basinhopping(func=loss_function, x0=[X_guess,Y_guess], niter=400, T=0, minimizer_kwargs = {"args":(valid_geometry[0],valid_geometry[1],valid_geometry[2])})
    
#     inv_hessian = result.lowest_optimization_result.hess_inv.todense()
#     # inv_hessian = result.lowest_optimization_result.hess_inv  

#     print(inv_hessian)
#     det_inv_hessian = inv_hessian[0][0] * inv_hessian[1][1] - inv_hessian[0][1] * inv_hessian[1][0]
#     res_x.append(result.x[0])
#     res_y.append(result.x[1])
#     x_err.append(np.sqrt(inv_hessian[1][1]/det_inv_hessian))
#     y_err.append(np.sqrt(inv_hessian[0][0]/det_inv_hessian))

# res_x =  np.array(res_x)
# res_y = np.array(res_y)
# print("",np.mean(res_x)," +/- ",np.mean(x_err))
# print("",np.mean(res_y)," +/- ",np.mean(y_err))






