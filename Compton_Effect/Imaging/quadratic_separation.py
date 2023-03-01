import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import scipy.optimize as spo
import scipy.signal as ssp
# import umap
# import umap.plot
#import hdbscan
import itertools
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
import scipy.optimize as spo
from alive_progress import alive_bar
# from sklearn.cluster import Birch
hep.style.use("CMS")
method_ = "BFGS"


def indices(lst, item):
    return [i for i, x in enumerate(lst) if x == item]

def drop_preds(data,preds,val):
    index = []
    preds = indices(preds,val)
    index.append(preds)
    merged = list(itertools.chain(*index))
    new_index = np.sort(list(set(merged)))
    new_data = data.drop(new_index,axis=0)
    return new_data

def scatter_difference(coordinates: list, alpha: np.array, d:np.array, s:np.array):
    X, Y = coordinates
    theta = np.arctan2(Y,(X-s))
    phi = np.arctan2(Y,(d-X))
    inner_value = np.pi - alpha - theta - phi
    return np.abs(inner_value)**2

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


def loss_minimizer(alpha:np.array, d:np.array, s:np.array):
    alpha = np.array(alpha)
    s = np.array(s)
    d = np.array(d)
    X_guess = (d[0]-s[0])/2
    Y_guess = ((d[0]-s[0]))/(np.tan(alpha[0]))

    res_x = []
    res_y = []
    x_err = []
    y_err = []

    for i in range(0,4):
        # result = spo.basinhopping(func=loss_function, x0=[X_guess,Y_guess], niter=800, T=0, minimizer_kwargs = {"args":(alpha,d,s),"method":"Powell","bounds":([0,15],[0,15])})
        result = spo.basinhopping(func=loss_function, x0=[X_guess,Y_guess], niter=800, T=0, minimizer_kwargs = {"args":(alpha,d,s),"method":method_,"bounds":([0,15],[0,15])})


        # inv_hessian = result.lowest_optimization_result.hess_inv.todense()
        # det_inv_hessian = inv_hessian[0][0] * inv_hessian[1][1] - inv_hessian[0][1] * inv_hessian[1][0]

        res_x.append(result.x[0])
        res_y.append(result.x[1])
        # x_err.append(np.sqrt(inv_hessian[1][1]/det_inv_hessian))
        # y_err.append(np.sqrt(inv_hessian[0][0]/det_inv_hessian))

    res_x =  np.array(res_x)
    res_y = np.array(res_y)
    return [np.mean(res_x),np.mean(res_y)]

# Declare true geometry
x_1_true = 12
x_2_true = 10
y_1_true = 9
y_2_true = 4

X_bounds = [1,20]
Y_bounds = [1,10]
geometries = []
six_alpha_temp = []
six_s_temp = []
six_d_temp = []
six_label = []
six_x = []
six_y = []
two_alpha_temp = []
two_s_temp = []
two_d_temp = []
two_label = []
two_x = []
two_y = []
valid_geometry = []
for x in range(X_bounds[0],X_bounds[1]+1):
    for y in range(Y_bounds[0],Y_bounds[1]+1):
        for s in range(X_bounds[0],X_bounds[1]+1):
            for d in range(X_bounds[0]+1, X_bounds[1]):
                # alpha = alpha_calc(x,y,d,s)
                # geometries.append([s,d,alpha,x,y])
                # print(x)
                if (x == x_1_true) and (y==y_1_true):
                    valid_alpha = alpha_calc(x,y,d,s)
                    valid_geometry.append([x,y,d,s,valid_alpha])
                    six_d_temp.append(d)
                    six_s_temp.append(s)
                    six_alpha_temp.append(valid_alpha)
                    six_label.append(6) #Add dynamic (X,Y) here
                    six_x.append(x)
                    six_y.append(y)
                if (x == x_2_true) and (y==y_2_true):
                    valid_alpha = alpha_calc(x,y,d,s)
                    valid_geometry.append([x,y,d,s,valid_alpha])
                    two_d_temp.append(d)
                    two_s_temp.append(s)
                    two_alpha_temp.append(valid_alpha)
                    two_label.append(2) #Add dynamic (X,Y) here
                    two_x.append(x)
                    two_y.append(y)

two_x =  (two_d_temp[0]-two_s_temp[0])/2
six_x =  (six_d_temp[0]-six_s_temp[0])/2
two_y = ((two_d_temp[0]-two_s_temp[0]))/(np.tan(two_alpha_temp[0]))
six_y = ((six_d_temp[0]-six_s_temp[0]))/(np.tan(six_alpha_temp[0]))
print(f'length of two alpha {len(two_alpha_temp)}')
print(f'length of six alpha {len(six_alpha_temp)}')
combined_alpha = np.array(two_alpha_temp + six_alpha_temp)
print(f'length of combined alpha {len(combined_alpha)}')
combined_s = np.array(two_s_temp + six_s_temp)
combined_d = np.array(two_d_temp + six_d_temp)
combined_labels = six_label + two_label
combined_x = []
combined_y = []
# print("sss",combined_s)
# print("ddd",combined_d)
with alive_bar(len(combined_s)) as bar:
    for i in range(0,len(combined_s)):
        # print("Iteration ",i," of ", len(combined_s))

        x_guess = float((combined_d[i]+combined_s[i])/2)
        # y_guess = ((combined_d[i]+combined_s[i]))/(np.sin(combined_alpha[i]))
        # y_guess = float(x_guess+1)
        y_guess = np.abs(0.5*((combined_d[i]-combined_s[i]))/(np.tan(combined_alpha[0])))

        # print("X-Guess",x_guess)      
        # print("Y-Guess",y_guess)
        # x_guess = 5
        # y_guess = 5
        bounds = spo.Bounds(lb=[0,0],ub=[20,20])
        # result = spo.basinhopping(func=scatter_difference, niter=500, x0=list([x_guess,y_guess]), T=0, minimizer_kwargs = {"args":(combined_alpha[i],combined_d[i],combined_s[i]),"method":method_,"bounds":bounds})
        result = spo.basinhopping(func=scatter_difference, niter=40, x0=list([x_guess,y_guess]), T=0, minimizer_kwargs = {"args":(combined_alpha[i],combined_d[i],combined_s[i]),"method":method_,"bounds":((X_bounds[0], Y_bounds[0]),(X_bounds[1], Y_bounds[1]))})

        # result = spo.basinhopping(func=scatter_difference, niter=500, x0=[x_guess,y_guess], T=0, minimizer_kwargs = {"args":(combined_alpha[i],combined_d[i],combined_s[i]),"method":'Powell',"bounds":([0,20],[0, 20])})

        if result.x[0] < 0:
            combined_x.append(0)
        else:
            combined_x.append(result.x[0])
        if result.x[1] < 0:
            combined_y.append(0)
        else:
            combined_y.append(result.x[1])
        bar()
combined_y = np.array(combined_y)
combined_x = np.array(combined_x)
combined_x = combined_x[combined_y > 1.5]
combined_y = combined_y[combined_y > 1.5]

plt.scatter(combined_x,combined_y)

combined_x, combined_y =  zip(*sorted(zip(combined_x, combined_y)))

combined_y = np.array(combined_y)
combined_x = np.array(combined_x)
number_of_intervals = 10

interval = int((len(combined_x)/number_of_intervals))
print(f'Interval {interval}')
temp_y = []
temp_x = []
for i in range(0,len(combined_x)-interval,interval):
    #num_of_bins = 20
    combined_y_bound = combined_y[(combined_x[i] < combined_x) & (combined_x < combined_x[i+interval])]
    combined_x_bound = combined_x[(combined_x[i] < combined_x) & (combined_x < combined_x[i+interval])]

    upper = np.percentile(combined_y_bound,90)
    lower = np.percentile(combined_y_bound,10)
    # combined_x_bound = combined_x_bound[(combined_y_bound < max(combined_y_bound)) & (combined_y_bound > min(combined_y_bound))]
    # combined_y_bound = combined_y_bound[(combined_y_bound < max(combined_y_bound)) & (combined_y_bound > min(combined_y_bound))]

    combined_x_bound = combined_x_bound[(combined_y_bound < upper) & (combined_y_bound > lower)]
    combined_y_bound = combined_y_bound[(combined_y_bound < upper) & (combined_y_bound > lower)]
    temp_x.append((combined_x[i+interval] + combined_x[i])/2)

    iqr = np.percentile(combined_y_bound,75) - np.percentile(combined_y_bound,25)
    #h = 2 * iqr / (len(combined_y_bound)**(1/2))
    num_of_bins = int((max(combined_y_bound) - min(combined_y_bound))/ 0.25)
    print(f'num_of_bins{num_of_bins}')
    hist_full, bin_edges_full = np.histogram(combined_y_bound, bins=num_of_bins)

    width = 0.1

    spacing_bin_size = width
    o = 1
    while (spacing_bin_size < 0.5):
        spacing_bin_size = width*o
        o += 1

    o = 2
    print(f'o {o}')
    print(f'width {width}')
    zero = True
    for k in range(len(hist_full)-(o-1)):
        zero = True
        print(hist_full[k:k+o])
        for x in hist_full[k:k+o]:
            if x != 0:
                zero = False
                break
        if zero == True:
            break

    if zero == True:
            print('Two POINTS')
    else:
        print('One POINTS')

    mid_y_point = (bin_edges_full[0] + bin_edges_full[-1])/2

    bh_y_bound = combined_y_bound[mid_y_point-0.25 > combined_y_bound]
    th_y_bound = combined_y_bound[mid_y_point+0.25 < combined_y_bound]
    plt.figure(1)
    plt.hist(combined_y_bound, bins = num_of_bins)
    plt.axvline(mid_y_point-0.25, color = 'black')
    plt.axvline(mid_y_point+0.25, color = 'black')

    # print(f'combined_x[i] {combined_x[i]}')
    # print(f'combined_x[i+interval] {combined_x[i+interval]}')
    # print(f'i {i}')
    # print(f'i+interval{i+interval}')
    plt.figure(2)
    plt.scatter(combined_x,combined_y)
    plt.scatter(combined_x_bound,combined_y_bound, color = 'red')
    plt.axvline(combined_x[i], color = 'black')
    plt.axvline(combined_x[i+interval], color = 'orange')
    plt.show()


plt.figure(3)
plt.scatter(combined_x,combined_y)
#plt.scatter(temp_x,temp_y)
plt.show()

