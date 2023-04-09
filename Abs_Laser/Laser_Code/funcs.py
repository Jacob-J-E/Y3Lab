import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
import scipy.interpolate as spi
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
from sklearn.preprocessing import *
from sklearn.cluster import DBSCAN
from itertools import chain
import copy
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from scipy.signal import butter, filtfilt
from skimage.restoration import richardson_lucy
from scipy.signal import convolve, gaussian, fftconvolve, wiener
from scipy.optimize import minimize
from scipy.ndimage import convolve1d
from alive_progress import alive_bar


R_1_R_2 = 0.995*0.995
F = 0.5 * (4 * (R_1_R_2)**(1/4))/(1-(R_1_R_2)**(1/2))
fsr_theoretical = (3e8/(4*20e-2))


def lorentzian(x, center, width, amplitude, c):
    """
    Calculate the Lorentzian function for the given x values.

    Parameters
    ----------
    x : array-like
        Input x values to calculate the Lorentzian function.
    center : float
        The center of the Lorentzian function.
    width : float
        The width of the Lorentzian function (also known as FWHM: Full Width at Half Maximum).
    amplitude : float
        The amplitude of the Lorentzian function.
    c: float
        Global vertical shift.

    Returns
    -------
    y : array-like
        The Lorentzian function values corresponding to the input x values.
    """
    y = ((amplitude / np.pi) * (width / 2) / ((x - center)**2 + (width / 2)**2))+ c
    return y

def lorentzian_normalised(x, center, width, amplitude, c):
    """
    Calculate the Lorentzian function for the given x values.

    Parameters
    ----------
    x : array-like
        Input x values to calculate the Lorentzian function.
    center : float
        The center of the Lorentzian function.
    width : float
        The width of the Lorentzian function (also known as FWHM: Full Width at Half Maximum).
    amplitude : float
        The amplitude of the Lorentzian function.
    c: float
        Global vertical shift.

    Returns
    -------
    y : array-like
        The Lorentzian function values corresponding to the input x values.
    """
    y = ((width / 2) / ((x - center)**2 + (width / 2)**2))+ c
    y = (y/max(y))*amplitude
    return y


def f(x,*coeff):
    coeff = list(coeff)
    x = np.array(x)
    function = np.zeros(len(x))
    for i in range(len(coeff)):
        function += coeff[i]*x**(i)
    return function

def airy_modified_function(x,*args):
    args = list(args)
    num = args[-2] + x*args[-1]
    f_values = f(x,*args[:-2])
    dom = 1 + F*(np.sin((np.pi/fsr_theoretical)*f_values))**2
    return (num/dom) 


def straight_line(x,a,b):
    return a*x+b

def five_lor_x(x,f1,w1,a1,f2,w2,a2,f3,w3,a3,f4,w4,a4,f5,w5,a5, a,b,c):
    return lorentzian(x,f1,w1,a1,0) + lorentzian(x,f2,w2,a2,0) + lorentzian(x,f3,w3,a3,0) + lorentzian(x,f4,w4,a4,0) + lorentzian(x,f5,w5,a5,0) - (a*x**2 + b*x) + c

def five_lor_x_update(x,f1,w1,a1,f2,w2,a2,f3,w3,a3,f4,w4,a4,f5,w5,a5,f6,w6,a6,m,c):
    return lorentzian(x,f1,w1,a1,0) + lorentzian(x,f2,w2,a2,0) + lorentzian(x,f3,w3,a3,0) + lorentzian(x,f4,w4,a4,0) + lorentzian(x,f5,w5,a5,0) +  lorentzian(x,f6,w6,a6,0) + straight_line(x,m,c)

def four_lor_x_update(x,f1,w1,a1,f2,w2,a2,f3,w3,a3,f4,w4,a4,m,c):
    return lorentzian_normalised(x,f1,w1,a1,0) + lorentzian_normalised(x,f2,w2,a2,0) + lorentzian_normalised(x,f3,w3,a3,0) + lorentzian_normalised(x,f4,w4,a4,0) + straight_line(x,m,c)
