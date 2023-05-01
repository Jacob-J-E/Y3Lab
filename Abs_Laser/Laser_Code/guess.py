import scipy.optimize as spo


spo.curve_fit(f=func,xdata=x,ydata=y,po=guess)