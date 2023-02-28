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
from sklearn.mixture import GaussianMixture
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


# CHANGE TO YOUR VALUE
# Declare true geometry
# x_1_true = 12
# x_2_true = 4
# y_1_true = 8
# y_2_true = 9
x_1_true = 20
x_2_true = 10
y_1_true = 9
y_2_true = 4

# CHANGE TO YOUR VALUE
X_bounds = [1,30]
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
combined_alpha = np.array(two_alpha_temp + six_alpha_temp)
combined_s = np.array(two_s_temp + six_s_temp)
combined_d = np.array(two_d_temp + six_d_temp)
combined_labels = six_label + two_label
combined_x = []
combined_y = []


for i in range(0,len(combined_s)):

    x_guess = float((combined_d[i]+combined_s[i])/2)
    # y_guess = ((combined_d[i]+combined_s[i]))/(np.sin(combined_alpha[i]))

    y_guess = np.abs(0.5*((combined_d[i]-combined_s[i]))/(np.tan(combined_alpha[0])))

    bounds = spo.Bounds(lb=[0,0],ub=[20,20])
    # result = spo.basinhopping(func=scatter_difference, niter=500, x0=list([x_guess,y_guess]), T=0, minimizer_kwargs = {"args":(combined_alpha[i],combined_d[i],combined_s[i]),"method":method_,"bounds":bounds})
    result = spo.basinhopping(func=scatter_difference, niter=20, x0=list([x_guess,y_guess]), T=0, minimizer_kwargs = {"args":(combined_alpha[i],combined_d[i],combined_s[i]),"method":method_,"bounds":([0,20],[0,20])})

    # result = spo.basinhopping(func=scatter_difference, niter=500, x0=[x_guess,y_guess], T=0, minimizer_kwargs = {"args":(combined_alpha[i],combined_d[i],combined_s[i]),"method":'Powell',"bounds":([0,20],[0, 20])})

    if result.x[0] < 0:
        combined_x.append(0)
    else:
        combined_x.append(result.x[0])
    if result.x[1] < 0:
        combined_y.append(0)
    else:
        combined_y.append(result.x[1])

data = {'x':combined_x,'y':combined_y}
data = pd.DataFrame(data = data)

# birch clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import Birch
from matplotlib import pyplot
# define dataset
# X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
X = np.array(data)
print("AHHH",len(X))
# X[0] = X[ : ,0][(X[1])>2]
# X[1] = X[ : ,1][(X[1])>2]
X = X[X[ : ,1] > 2]

# X.remove(X[1][X[1] < 2])
print("X data",X)
# define the model
model = GaussianMixture(n_components=4,n_init=4,max_iter=500)
# fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
coordinates = []
for i,cluster in enumerate(clusters):
 # get row indexes for samples with this cluster
 row_ix = where(yhat == cluster)
 # create scatter of these samples
 pyplot.scatter(X[row_ix, 0], X[row_ix, 1],label="Cluster "+str(i))
 coordinates.append(np.array([X[row_ix, 0], X[row_ix, 1]]))
# show the plot
# print(coordinates)
pyplot.xlabel("X position (arb.)")
pyplot.ylabel("Y position (arb.)")
pyplot.legend(loc="upper right")
pyplot.show()

for i in range(0,len(coordinates)):
    plt.scatter(coordinates[i][0],coordinates[i][1])

    plt.show()
    d = {'x':np.array(coordinates[i][0][0]),'y':np.array(coordinates[i][1][0])}
    Y = pd.DataFrame(d)
    Y = np.array(Y)

    # define the model
    model = GaussianMixture(n_components=4,n_init=4,max_iter=500)
    # fit the model
    model.fit(Y)
    # assign a cluster to each example
    yhat = model.predict(Y)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    # coordinates = []
    for j,cluster in enumerate(clusters):
     # get row indexes for samples with this cluster
     row_ix_ = where(yhat == cluster)
     # create scatter of these samples
     pyplot.scatter(Y[row_ix_, 0], Y[row_ix_, 1],label="Cluster "+str(j))
     # show the plot
 
    pyplot.xlabel("X position (arb.)")
    pyplot.ylabel("Y position (arb.)")
    pyplot.legend(loc="upper right")
    
    
    pyplot.show()






