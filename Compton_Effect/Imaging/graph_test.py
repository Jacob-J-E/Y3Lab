import seaborn as sns
sns.set_theme(style="ticks")
from sklearn.mixture import GaussianMixture
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

def gaussian(x, a, b, c):
    return (a * np.exp(-((x - b) ** 2) / (2 * c ** 2)))

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
y_2_true = 8

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

np.savetxt("combined_x.csv", combined_x, delimiter=",")
np.savetxt("combined_y.csv", combined_y, delimiter=",")
np.savetxt("combined_alpha.csv", combined_y, delimiter=",")
np.savetxt("combined_d.csv", combined_y, delimiter=",")
np.savetxt("combined_s.csv", combined_y, delimiter=",")
d  = {"x":combined_x,"y":combined_y}
data = pd.DataFrame(d)

# Load the planets dataset and initialize the figure
# planets = sns.load_dataset("planets")

# g = sns.JointGrid(data=data, x="x", y="y", marginal_ticks=True)

# # Set a log scaling on the y axis
# # g.ax_joint.set(yscale="log")

# # Create an inset legend for the histogram colorbar
# cax = g.figure.add_axes([.15, .55, .02, .2])

# # Add the joint and marginal histogram plots
# g.plot_joint(
#     sns.histplot, discrete=(False, False),
#     cmap="light:#03012d", pmax=.8, cbar=True, cbar_ax=cax, kde=False, stat='probability', log_scale=False, cumulative=False
# )
# g.plot_marginals(sns.histplot, element="step", color="#03012d")



# plt.show()

# iqr = np.percentile(combined_y,75) - np.percentile(combined_y,25)
# h = 2 * iqr / (len(combined_y)**(1/3))
# num_of_bins = int((max(combined_y) - min(combined_y))/ h)
# print(f'num_of_bins{num_of_bins}')
# hist_full, bin_edges_full = np.histogram(combined_y, bins=num_of_bins)
# centers = (bin_edges_full[1:] + bin_edges_full[:-1])/2
# print(f'hist_full {hist_full}')
# print(f'width {bin_edges_full[1] - bin_edges_full[0]}')
# width = bin_edges_full[1] - bin_edges_full[0]
# # plt.hist(combined_y, bins=num_of_bins)
# # plt.scatter(centers,hist_full)
# hist_full_sort,centers_sort = zip(*sorted(zip(hist_full,centers),reverse=True))
# print(centers_sort)
# full_dataset_centers = centers_sort
# plt.figure()
# plt.scatter(combined_x,combined_y)
# plt.axhline(centers_sort[0]+width/2, color = 'black')
# plt.axhline(centers_sort[0]-width/2, color = 'black')
# plt.axhline(centers_sort[1]+width/2, color = 'orange')
# plt.axhline(centers_sort[1]-width/2, color = 'orange')
# combined_x_higher = combined_x[(centers_sort[0]+width/2>combined_y) & (centers_sort[0]-width/2<combined_y)]
# combined_x_lower = combined_x[(centers_sort[1]+width/2>combined_y) & (centers_sort[1]-width/2<combined_y)]


# def gaussian(x, a, b, c):
#     return (a * np.exp(-((x - b) ** 2) / (2 * c ** 2)))

# plt.figure()
# iqr = np.percentile(combined_x_higher,75) - np.percentile(combined_x_higher,25)
# h = 2 * iqr / (len(combined_x_higher)**(1/3))
# num_of_bins = int((max(combined_x_higher) - min(combined_x_higher))/ h)

# hist_full, bin_edges_full = np.histogram(combined_x_higher, bins=num_of_bins)
# centers = (bin_edges_full[1:] + bin_edges_full[:-1])/2
# hist_full_sort,centers_sort = zip(*sorted(zip(hist_full,centers),reverse=True))

# guess_e1 = [hist_full_sort[0],centers_sort[0], 2]
# params_e1, cov_e1 = spo.curve_fit(gaussian,centers_sort,hist_full_sort,guess_e1)

# domain = np.linspace(min(combined_x_higher),max(combined_x_higher),10000)
# plt.hist(combined_x_higher, bins=num_of_bins)
# plt.plot(domain,gaussian(domain,*params_e1))
# print(f'params_e1 {params_e1}')


# plt.figure()
# iqr = np.percentile(combined_x_lower,75) - np.percentile(combined_x_lower,25)
# h = 2 * iqr / (len(combined_x_lower)**(1/3))
# num_of_bins = int((max(combined_x_lower) - min(combined_x_lower))/ h)

# hist_full, bin_edges_full = np.histogram(combined_x_lower, bins=num_of_bins)
# centers = (bin_edges_full[1:] + bin_edges_full[:-1])/2
# hist_full_sort,centers_sort = zip(*sorted(zip(hist_full,centers),reverse=True))

# guess_e2 = [hist_full_sort[0],centers_sort[0], 2]
# params_e2, cov_e2 = spo.curve_fit(gaussian,centers_sort,hist_full_sort,guess_e2)

# domain = np.linspace(min(combined_x_lower),max(combined_x_lower),10000)
# plt.hist(combined_x_lower, bins=num_of_bins)
# plt.plot(domain,gaussian(domain,*params_e2))
# print(f'params_e2 {params_e2}')

# print(f'x1: {params_e1[1]} | y1: {full_dataset_centers[0]}')
# print(f'x2: {params_e2[1]} | y2: {full_dataset_centers[1]}')
 

# plt.show()

 

def check_values(x_data, y_data):
    combined_x = x_data
    combined_y = y_data
    iqr = np.percentile(combined_y,75) - np.percentile(combined_y,25)
    h = 2 * iqr / (len(combined_y)**(1/3))
    num_of_bins = int((max(combined_y) - min(combined_y))/ h)
    #print(f'num_of_bins{num_of_bins}')
    hist_full, bin_edges_full = np.histogram(combined_y, bins=num_of_bins)
    centers = (bin_edges_full[1:] + bin_edges_full[:-1])/2
    # print(f'hist_full {hist_full}')
    # print(f'width {bin_edges_full[1] - bin_edges_full[0]}')
    width = bin_edges_full[1] - bin_edges_full[0]
    # plt.hist(combined_y, bins=num_of_bins)
    # plt.scatter(centers,hist_full)
    hist_full_sort,centers_sort = zip(*sorted(zip(hist_full,centers),reverse=True))
    #print(centers_sort)
    full_dataset_centers = centers_sort
    # plt.figure()
    # plt.scatter(combined_x,combined_y)
    # plt.axhline(centers_sort[0]+width/2, color = 'black')
    # plt.axhline(centers_sort[0]-width/2, color = 'black')
    # plt.axhline(centers_sort[1]+width/2, color = 'orange')
    # plt.axhline(centers_sort[1]-width/2, color = 'orange')
    combined_x_higher = combined_x[(centers_sort[0]+width/2>combined_y) & (centers_sort[0]-width/2<combined_y)]
    combined_x_lower = combined_x[(centers_sort[1]+width/2>combined_y) & (centers_sort[1]-width/2<combined_y)]


    #plt.figure()
    iqr = np.percentile(combined_x_higher,75) - np.percentile(combined_x_higher,25)
    h = 2 * iqr / (len(combined_x_higher)**(1/3))
    num_of_bins = int((max(combined_x_higher) - min(combined_x_higher))/ h)

    hist_full, bin_edges_full = np.histogram(combined_x_higher, bins=num_of_bins)
    centers = (bin_edges_full[1:] + bin_edges_full[:-1])/2
    hist_full_sort,centers_sort = zip(*sorted(zip(hist_full,centers),reverse=True))

    guess_e1 = [hist_full_sort[0],centers_sort[0], 2]

    try:
        params_e1, cov_e1 = spo.curve_fit(gaussian,centers_sort,hist_full_sort,guess_e1)
    except:
        params_e1 = [-1,-1,-1]

    #params_e1, cov_e1 = spo.curve_fit(gaussian,centers_sort,hist_full_sort,guess_e1)

    domain = np.linspace(min(combined_x_higher),max(combined_x_higher),10000)
    # plt.hist(combined_x_higher, bins=num_of_bins)
    # plt.plot(domain,gaussian(domain,*params_e1))
    # print(f'params_e1 {params_e1}')


    #plt.figure()
    iqr = np.percentile(combined_x_lower,75) - np.percentile(combined_x_lower,25)
    h = 2 * iqr / (len(combined_x_lower)**(1/3))
    num_of_bins = int((max(combined_x_lower) - min(combined_x_lower))/ h)

    hist_full, bin_edges_full = np.histogram(combined_x_lower, bins=num_of_bins)
    centers = (bin_edges_full[1:] + bin_edges_full[:-1])/2
    hist_full_sort,centers_sort = zip(*sorted(zip(hist_full,centers),reverse=True))

    guess_e2 = [hist_full_sort[0],centers_sort[0], 2]
    try:
        params_e2, cov_e2 = spo.curve_fit(gaussian,centers_sort,hist_full_sort,guess_e2)
    except:
        params_e2 = [-1,-1,-1]


                                
    domain = np.linspace(min(combined_x_lower),max(combined_x_lower),10000)
    # plt.hist(combined_x_lower, bins=num_of_bins)
    # plt.plot(domain,gaussian(domain,*params_e2))
    # print(f'params_e2 {params_e2}')

    # print(f'x1: {params_e1[1]} | y1: {full_dataset_centers[0]}')
    # print(f'x2: {params_e2[1]} | y2: {full_dataset_centers[1]}')

    return params_e1[1],params_e2[1],full_dataset_centers[0],full_dataset_centers[1]

# x1,x2,y1,y2 = check_values(combined_x,combined_y)
# print(x1,x2,y1,y2)




# model = GaussianMixture(n_components=4,n_init=4,max_iter=500)
# # fit the model
# model.fit(clusterable_embedding)
# # assign a cluster to each example
# yhat = model.predict(clusterable_embedding)
# # retrieve unique clusters
# clusters = np.unique(yhat)
# # create scatter plot for samples from each cluster
# coordinates = []
# preds = []
# for i,cluster in enumerate(clusters):
#  # get row indexes for samples with this cluster
#  row_ix = np.where(yhat == cluster)
#  # create scatter of these samples
#  plt.scatter(clusterable_embedding[row_ix, 0], clusterable_embedding[row_ix, 1],label="Cluster "+str(i))
#  coordinates.append([clusterable_embedding[row_ix, 0], clusterable_embedding[row_ix, 1]])
#  preds.append(row_ix)
# # show the plot

# marker = []
# for i in range(len(X_train)):
#     if i in preds[0][0]:
#         marker.append(0)
    
#     elif i in preds[1][0]:
#         marker.append(1)

# X_train['marker'] = marker

# cluster_0 = X_train[X_train['marker'] == 0]
# cluster_1 = X_train[X_train['marker'] == 1]


