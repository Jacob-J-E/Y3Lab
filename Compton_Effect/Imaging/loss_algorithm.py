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


# import numpy as np
# from scipy.optimize import minimize

# define your 2D function
def neg_log_likelihood(x1, x2):
    x = np.array([x1, x2])  # combine x1 and x2 into an array
    mu = np.array([4, 5])  # mean vector
    cov = np.array([[1, 0.5], [0.5, 2]])  # covariance matrix
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    diff = x - mu
    exponent = -0.5 * np.dot(np.dot(diff, inv), diff)
    return -np.log(2*np.pi*det) - 0.5*exponent

# define initial guess
x0 = np.array([0, 0])

# minimize the negative log likelihood
result = spo.minimize(lambda x: neg_log_likelihood(x[0], x[1]), x0)

# print the optimized result
print(result)
