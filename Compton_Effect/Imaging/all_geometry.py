import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import scipy.optimize as spo
import scipy.signal as ssp
hep.style.use("CMS")


def alpha_calc(X,Y,d,s):
    theta = np.arctan2(Y,(X-s))
    phi = np.arctan2(Y,(d-X))
    alpha = np.pi - theta - phi
    return alpha

def geo_difference(theory,exp):
    diff = np.sqrt(np.sum((theory-exp)**2))
    return diff



 

X_bounds = [5,30]
Y_bounds = [1,20]
geometries = []

alpha_temp = []
s_temp = []
d_temp = []

valid_geometry = []
for x in range(X_bounds[0],X_bounds[1]+1):
    for y in range(Y_bounds[0],Y_bounds[1]+1):
        for s in range(1,X_bounds[0]+10):
            for d in range(X_bounds[1]+1, 30+10):
                # alpha = alpha_calc(x,y,d,s)
                # geometries.append([s,d,alpha,x,y])
                # print(x)
                if (x == 6) and (y==4):
                    print("AHh")
                    valid_alpha = alpha_calc(x,y,d,s)
                    valid_geometry.append([x,y,d,s,valid_alpha])
                    d_temp.append(d)
                    s_temp.append(s)
                    alpha_temp.append(valid_alpha)


print(d_temp)
print('-------')
print(s_temp)
print('-------')
print(alpha_temp)
print('-------')


# exp_geo = []