import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import scipy.optimize as spo
import scipy.signal as ssp
import umap
import umap.plot
import hdbscan
import itertools
from sklearn.cluster import Birch
import matplotlib.colors as colors
hep.style.use("CMS")
method_ = "BFGS"
from matplotlib.colors import LogNorm


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
x_2_true = 4
y_1_true = 5
y_2_true = 3

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
                if (x == x_1_true) and (y==y_2_true):
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
# print(f'length of two alpha {len(two_alpha_temp)}')
# print(f'length of six alpha {len(six_alpha_temp)}')
combined_alpha = np.array(two_alpha_temp + six_alpha_temp)
# print(f'length of combined alpha {len(combined_alpha)}')
combined_s = np.array(two_s_temp + six_s_temp)
combined_d = np.array(two_d_temp + six_d_temp)
combined_labels = six_label + two_label
combined_x = []
combined_y = []
# print("sss",combined_s)
# print("ddd",combined_d)

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
    result = spo.basinhopping(func=scatter_difference, niter=40, x0=list([x_guess,y_guess]), T=0, minimizer_kwargs = {"args":(combined_alpha[i],combined_d[i],combined_s[i]),"method":method_,"bounds":([0,20],[0,20])})

    # result = spo.basinhopping(func=scatter_difference, niter=500, x0=[x_guess,y_guess], T=0, minimizer_kwargs = {"args":(combined_alpha[i],combined_d[i],combined_s[i]),"method":'Powell',"bounds":([0,20],[0, 20])})

    if result.x[0] < 0:
        combined_x.append(0)
    else:
        combined_x.append(result.x[0])
    if result.x[1] < 0:
        combined_y.append(0)
    else:
        combined_y.append(result.x[1])


# plt.scatter(combined_x,combined_y)
# plt.show()
data = {'x':combined_x,'y':combined_y}
data = pd.DataFrame(data = data)



birch = Birch(n_clusters=2)
fit = birch.fit(data)
plt.scatter(data['x'],data['y'],c=birch.labels_)
plt.show()
# birch = Birch(n_clusters=2)
# fit = birch.fit(data)
# plt.scatter(data['x'],data['y'],c=birch.labels_)
# plt.show()


fig, axes = plt.subplots(nrows=1, ncols=1)

"""
Gaussian KDE Method
"""
import seaborn as sns

# A = sns.histplot(data=data,x='x',y='y',stat='density',bins=20,cbar=True,cmap='inferno',thresh=None, norm=LogNorm(), vmin=1, vmax=1e2,)             # use for log color bar
A = sns.histplot(data=data,x='x',y='y',stat='frequency',bins=100,cbar=True,cmap='inferno',thresh=30,kde=0)             # use for log color bar
B = A.patches
print(B)
C = [A.get_height() for A in A.patches]
print(C)
aspect_ratio = 0.7
x_left, x_right = A.get_xlim()
y_left, y_right = A.get_ylim()
axes.set_aspect(np.abs((x_right-x_left)/(y_left-y_right))*aspect_ratio)
fig.tight_layout()
plt.show()

# generate data from loaded .csv
# x = np.array(data['x'])
# y = np.array(data['y'])
# data = np.vstack((x,y))
# import scipy as sp
# #generate Gaussian KDE
# X, Y = np.mgrid[min(x):max(x):170j, min(y):max(y):170j]
# positions = np.vstack([X.ravel(), Y.ravel()])
# kernel = sp.stats.gaussian_kde(data)
# Z = np.reshape(kernel(positions).T, X.shape)
# x = np.linspace(min(y),max(y),5000)

# A = kernel.pdf(data)
# print(A)
# fig, ax = plt.subplots(1,1)
# contour = ax.contourf(X,Y,Z,levels=2000,cmap='inferno')#,norm=LogNorm(), vmin=1, vmax=1e2)
# cbar = fig.colorbar(contour)
# print(np.shape(A))
# print(np.shape(y))
# ax.set_xlabel(r"$B_0$ Mass $[MeV/c^2]$")
# ax.set_ylabel(r"$q^2$ Mass $[MeV/c^2]$")
# plt.show()
