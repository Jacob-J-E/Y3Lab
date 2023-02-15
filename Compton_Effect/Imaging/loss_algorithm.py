import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import scipy.optimize as spo
import scipy.signal as ssp
hep.style.use("CMS")

def scatter_deviation(alpha: np.array, d:np.array, s:np.array, X: float, Y: float):
    theta = np.arctan2(Y/(X-s))
    phi = np.arctan2(X/(d-X))
    inner_value = np.pi - alpha - theta - phi
    return np.abs(inner_value)**2

def loss(f):
    return np.sum(f)

def loss_function(coordinates: list, alpha: np.array, d:np.array, s:np.array):
    X, Y = coordinates
    theta = np.arctan2(Y,(X-s))
    phi = np.arctan2(X,(d-X))
    inner_value = np.pi - alpha - theta - phi
    return np.sum(np.abs(inner_value)**2)
    # return np.abs(inner_value)**2


alpha_test = np.array([0,1,0,0])
d_test = np.array([0,0,0,0])
s_test = np.array([0,0,0,0])

result = spo.minimize(fun=loss_function, x0=[0,0], args=(alpha_test,d_test,s_test))

# print the optimized result
print(result)
