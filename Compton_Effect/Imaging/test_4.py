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


combined_x = np.loadtxt("Compton_Effect/Imaging/combined_x.csv", delimiter=",")
combined_y = np.loadtxt("Compton_Effect/Imaging/combined_y.csv", delimiter=",")
combined_d = np.loadtxt("Compton_Effect/Imaging/combined_d.csv", delimiter=",")
combined_s = np.loadtxt("Compton_Effect/Imaging/combined_s.csv", delimiter=",")
combined_alpha = np.loadtxt("Compton_Effect/Imaging/combined_alpha.csv", delimiter=",")



def loss_function(coordinates: list, alpha: np.array, d:np.array, s:np.array):
    alpha = np.array(alpha)
    d = np.array(d)
    s = np.array(s)
    X, Y = coordinates
    theta = np.arctan2(Y,(X-s))
    phi = np.arctan2(Y,(d-X))
    # inner_value = (np.pi - alpha - theta - phi)*percentage_difference
    inner_value = (np.pi - alpha - theta - phi)

    return np.sum(np.abs(inner_value)**2)

def loss_minimizer(alpha:np.array, d:np.array, s:np.array):
    alpha = np.array(alpha)
    s = np.array(s)
    d = np.array(d)
    X_guess = (d[0]-s[0])/2
    Y_guess = ((d[0]-s[0]))/(np.tan(alpha[0]))

    X_guess = 11.5
    Y_guess = 6

    res_x = []
    res_y = []
    x_err = []
    y_err = []

    for i in range(0,4):
        result = spo.basinhopping(func=loss_function, x0=[X_guess,Y_guess], niter=200, T=0, minimizer_kwargs = {"args":(alpha,d,s),"method":method_})

        # inv_hessian = result.lowest_optimization_result.hess_inv.todense()
        inv_hessian = result.lowest_optimization_result.hess_inv

        det_inv_hessian = inv_hessian[0][0] * inv_hessian[1][1] - inv_hessian[0][1] * inv_hessian[1][0]

        res_x.append(result.x[0])
        res_y.append(result.x[1])
        # x_err.append(np.sqrt(inv_hessian[1][1]/det_inv_hessian))
        # y_err.append(np.sqrt(inv_hessian[0][0]/det_inv_hessian))
        x_err.append(np.sqrt(inv_hessian[0][0]))
        y_err.append(np.sqrt(inv_hessian[1][1]))

    res_x =  np.array(res_x)
    res_y = np.array(res_y)
    x_err = np.mean(np.abs(np.array(x_err)))
    y_err = np.mean(np.abs(np.array(y_err)))

    return [np.mean(res_x),np.mean(res_y),3*x_err,3*y_err]


def gaussian(x, a, b, c):
    return (a * np.exp(-((x - b) ** 2) / (2 * c ** 2)))


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

def cluster_gaus(x,y,n_clusters):
    combined_x = x
    combined_y = y
    data = {'x':combined_x,'y':combined_y} 
    data = np.array(pd.DataFrame(data))
    X_train = data
    X = X_train

    model = GaussianMixture(n_components=n_clusters,n_init=4,max_iter=500,random_state=42)
    
    # fit the model
    model.fit(X)

    # assign a cluster to each example
    yhat = model.predict(X)

    # retrieve unique clusters
    clusters = np.unique(yhat)
    coordinates = []
    preds = []
    for i,cluster in enumerate(clusters):
        # get row indexes for samples with this cluster
        row_ix = np.where(yhat == cluster)
        
        coordinates.append([X[row_ix, 0], X[row_ix, 1]])
        preds.append(row_ix)

    return coordinates,preds


combined_x = np.loadtxt("Compton_Effect/Imaging/combined_x.csv", delimiter=",")
combined_y = np.loadtxt("Compton_Effect/Imaging/combined_y.csv", delimiter=",")
combined_d = np.loadtxt("Compton_Effect/Imaging/combined_d.csv", delimiter=",")
combined_s = np.loadtxt("Compton_Effect/Imaging/combined_s.csv", delimiter=",")
combined_alpha = np.loadtxt("Compton_Effect/Imaging/combined_alpha.csv", delimiter=",")
data = {'x':combined_x,'y':combined_y,'combined_alpha':combined_alpha,'combined_s':combined_s, 'combined_d':combined_d} 
data = pd.DataFrame(data)
X_train = data

x1_i,x2_i,y1_i,y2_i = check_values(combined_x,combined_y)
print(x1_i,x2_i,y1_i,y2_i)
coordinates,preds = cluster_gaus(combined_x,combined_y,n_clusters=3)


def condition_checker(x1_t,x2_t,y1_t,y2_t):
    if x1_t != -1 and x2_t != -1:
        if abs(x1_t - x1_i) < 1.5 and abs(x2_t - x2_i) < 1.5 and abs(y1_t - y1_i) < 1.5 and abs(y2_t - y2_i) < 1.5:
            return 'contains_both'
        elif abs(x1_t - x1_i) < 1 and abs(y1_t - y1_i) < 1:
            return 'contains_xy1'
        elif abs(x2_t - x2_i) < 1.5 and abs(y2_t - y2_i) < 1.5:
            return 'contains_xy2'
        else:
            return 'no_converge'
    else:
        return 'no_converge'
    
xy1,xy2 = False, False
xy_clusters = []
# def finding_clusters(coordinates):
#     for i in range(len(coordinates)):
#         x1_t,x2_t,y1_t,y2_t = check_values(np.array(coordinates[i][0][0]),np.array(coordinates[i][1][0]))
#         condition = condition_checker(x1_t,x2_t,y1_t,y2_t )
#         if condition == 'contains_both':
#             print('contains_both')
#             sub_coordinates,sub_preds = cluster_gaus(coordinates[i][0][0],coordinates[i][1][0],n_clusters=4)
#             for i in range(len(sub_coordinates)):
#                 plt.scatter(sub_coordinates[i][0][0],sub_coordinates[i][1][0])
#             plt.show()
#             return finding_clusters(sub_coordinates)
#         elif condition == 'contains_xy1' and xy1 == False:
#             print('contains_xy1')
#             xy_clusters.append([np.array(coordinates[i][0][0]),np.array(coordinates[i][1][0])])
#             xy1 == True
#             if xy2 == True:
#                 return xy_clusters
#         elif condition == 'contains_xy2' and xy2 == False:
#             print('contains_xy2')
#             xy_clusters.append([np.array(coordinates[i][0][0]),np.array(coordinates[i][1][0])])
#             xy2 == True
#             if xy1 == True:
#                 return xy_clusters
#         else:
#             print('Hm')

xy_data = []
dataframe_clusters = []
def finding_clusters(coordinates):
    global clusters_array
    for i in range(len(coordinates)):
        x1_t,x2_t,y1_t,y2_t = check_values(np.array(coordinates[i][0][0]),np.array(coordinates[i][1][0]))
        condition = condition_checker(x1_t,x2_t,y1_t,y2_t )
        if condition == 'contains_both':
            print('contains_both')
            #plt.scatter(coordinates[i][0][0],coordinates[i][1][0])
            coordinates_sub,preds_sub = cluster_gaus(np.array(coordinates[i][0][0]),np.array(coordinates[i][1][0]),n_clusters=3)
            cluster_df = clusters_array[i]
            marker = []
            for i in range(len(cluster_df)):
                if i in preds_sub[0][0]:
                    marker.append(0)
                
                elif i in preds_sub[1][0]:
                    marker.append(1)

                elif i in preds_sub[2][0]:
                    marker.append(2)

            cluster_df['marker'] = marker

            cluster_0 = cluster_df[cluster_df['marker'] == 0]
            cluster_1 = cluster_df[cluster_df['marker'] == 1]
            cluster_2 = cluster_df[cluster_df['marker'] == 2]

            clusters_array = [cluster_0,cluster_1,cluster_2]
            for k in range(len(coordinates_sub)):
                x1_s,x2_s,y1_s,y2_s = check_values(np.array(coordinates_sub[k][0][0]),np.array(coordinates_sub[k][1][0]))
                condition = condition_checker(x1_s,x2_s,y1_s,y2_s)
                #plt.scatter(coordinates[i][0][0],coordinates[i][1][0])
                #plt.scatter(coordinates_sub[k][0][0],coordinates_sub[k][1][0])
                print(condition)
                #plt.show()
                if condition == 'contains_xy2':
                    #print('contains_both_sub')
                    xy_data.append([coordinates_sub[k][0][0],coordinates_sub[k][1][0]])
                    dataframe_clusters.append(clusters_array[k])
                    #plt.show()
                elif condition == 'no_converge' and xy1 == False:
                    #print('contains_xy1_sub')
                    xy_data.append([coordinates_sub[k][0][0],coordinates_sub[k][1][0]])
                    dataframe_clusters.append(clusters_array[k])
                    #plt.show()
        elif condition == 'contains_xy1' and xy1 == False:
            print('contains_xy1')
            # plt.scatter(coordinates[i][0][0],coordinates[i][1][0])
            # plt.show()
        elif condition == 'contains_xy2' and xy2 == False:
            print('contains_xy2')
            # plt.scatter(coordinates[i][0][0],coordinates[i][1][0])
            # plt.show()
        else:
            print('Hm')



marker = []
for i in range(len(X_train)):
    if i in preds[0][0]:
        marker.append(0)
    
    elif i in preds[1][0]:
        marker.append(1)

    elif i in preds[2][0]:
        marker.append(2)

X_train['marker'] = marker

cluster_0 = X_train[X_train['marker'] == 0]
cluster_1 = X_train[X_train['marker'] == 1]
cluster_2 = X_train[X_train['marker'] == 2]

clusters_array = [cluster_0,cluster_1,cluster_2]

finding_clusters(coordinates)
# plt.scatter(xy_data[0][0],xy_data[0][1])
# plt.scatter(xy_data[1][0],xy_data[1][1])
df_cluster1 = dataframe_clusters[0].drop(columns =['marker'])
df_cluster2 = dataframe_clusters[1].drop(columns =['marker'])
# print(df_cluster1)
# print(df_cluster2)

plt.scatter(df_cluster1['x'],df_cluster1['y'])
plt.scatter(df_cluster2['x'],df_cluster2['y'])

# print('----------------- GAUSSIAN MEAN ANGLE VALUES-----------------')
# for i in range(len(mean_angle_list)):
#     print(f'File: {list_files_sorted[i]}, Angle: {mean_angle_list[i]*(180/np.pi):.4g} +/- {standard_dev_angle_list[i]:.4g} ({(100*standard_dev_angle_list[i]/mean_angle_list[i]):.4g}%)')
print('-------------------------------------------------------------')
calculated_coordinates = loss_minimizer(df_cluster1['combined_alpha'],df_cluster1['combined_d'],df_cluster1['combined_s'])
print(f'X: {calculated_coordinates[0]:.4g} +/- {calculated_coordinates[2]:.4g} | Y: {calculated_coordinates[1]:.4g} +/- {calculated_coordinates[3]:.4g}')
print('-------------------------------------------------------------')
calculated_coordinates =  loss_minimizer(df_cluster2['combined_alpha'],df_cluster2['combined_d'],df_cluster2['combined_s'])
print(f'X: {calculated_coordinates[0]:.4g} +/- {calculated_coordinates[2]:.4g} | Y: {calculated_coordinates[1]:.4g} +/- {calculated_coordinates[3]:.4g}')

# plt.show()

