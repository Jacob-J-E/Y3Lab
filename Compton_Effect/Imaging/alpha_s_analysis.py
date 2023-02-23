import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import scipy.optimize as spo
import scipy.signal as ssp
import umap
import umap.plot
import hdbscan
hep.style.use("CMS")

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
        result = spo.basinhopping(func=loss_function, x0=[X_guess,Y_guess], niter=800, T=0, minimizer_kwargs = {"args":(alpha,d,s),"bounds":([0,15],[0,15])})
        
        inv_hessian = result.lowest_optimization_result.hess_inv.todense()
        det_inv_hessian = inv_hessian[0][0] * inv_hessian[1][1] - inv_hessian[0][1] * inv_hessian[1][0]
        
        res_x.append(result.x[0])
        res_y.append(result.x[1])
        x_err.append(np.sqrt(inv_hessian[1][1]/det_inv_hessian))
        y_err.append(np.sqrt(inv_hessian[0][0]/det_inv_hessian))

    res_x =  np.array(res_x)
    res_y = np.array(res_y)
    return [np.mean(res_x),np.mean(res_y)] 

 

X_bounds = [1,10]
Y_bounds = [1,10]
geometries = []

# six_alpha_temp = []
# six_s_temp = []
# six_d_temp = []
# six_label = []
# six_x = []
# six_y = []

# two_alpha_temp = []
# two_s_temp = []
# two_d_temp = []
# two_label = []
# two_x = []
# two_y = []

d9_valid_geometry = []
d9_six_d_temp = []
d9_six_s_temp = []
d9_six_alpha_temp = []
d9_six_label = []
d9_six_x = []
d9_six_y = []

d9_valid_alpha = []
valid_geometry= []
d9_two_d_temp= []
d9_two_s_temp= []
d9_two_alpha_temp= []
d9_two_label= []
d9_two_x= []
d9_two_y= []

d10_valid_alpha= []
d10_valid_geometry= []
d10_six_d_temp= []
d10_six_s_temp= []
d10_six_alpha_temp= []
d10_six_label= []
d10_six_x= []
d10_six_y= []

d10_valid_alpha = []
d10_valid_geometry= []
d10_two_d_temp= []
d10_two_s_temp= []
d10_two_alpha_temp= []
d10_two_label= []
d10_two_x= []
d10_two_y= []

valid_geometry = []
for x in range(X_bounds[0],X_bounds[1]+1):
    for y in range(Y_bounds[0],Y_bounds[1]+1):
        for s in range(X_bounds[0],X_bounds[1]+1):
            # for d in range(X_bounds[0]+1, X_bounds[1]):
            for d in range(9, 11):

                # alpha = alpha_calc(x,y,d,s)
                # geometries.append([s,d,alpha,x,y])
                # print(x)

                if d==9:
                    if (x == 9) and (y==4):
                        valid_alpha = alpha_calc(x,y,d,s)
                        # valid_geometry.append([x,y,d,s,valid_alpha])
                        # six_d_temp.append(d)
                        # six_s_temp.append(s)
                        # six_alpha_temp.append(valid_alpha)
                        # six_label.append(6)
                        # six_x.append(x)
                        # six_y.append(y)
                        d9_valid_alpha = alpha_calc(x,y,d,s)
                        d9_valid_geometry.append([x,y,d,s,valid_alpha])
                        d9_six_d_temp.append(d)
                        d9_six_s_temp.append(s)
                        d9_six_alpha_temp.append(valid_alpha)
                        d9_six_label.append(6)
                        d9_six_x.append(x)
                        d9_six_y.append(y)

                    if (x == 2) and (y==7):
                        valid_alpha = alpha_calc(x,y,d,s)
                        valid_geometry.append([x,y,d,s,valid_alpha])
                        d9_two_d_temp.append(d)
                        d9_two_s_temp.append(s)
                        d9_two_alpha_temp.append(valid_alpha)
                        d9_two_label.append(2)
                        d9_two_x.append(x)
                        d9_two_y.append(y)

                if d==10:
                    if (x == 9) and (y==4):
                        valid_alpha = alpha_calc(x,y,d,s)
                        d10_valid_geometry.append([x,y,d,s,valid_alpha])
                        d10_six_d_temp.append(d)
                        d10_six_s_temp.append(s)
                        d10_six_alpha_temp.append(valid_alpha)
                        d10_six_label.append(6)
                        d10_six_x.append(x)
                        d10_six_y.append(y)

                    if (x == 2) and (y==7):
                        valid_alpha = alpha_calc(x,y,d,s)
                        d10_valid_geometry.append([x,y,d,s,valid_alpha])
                        d10_two_d_temp.append(d)
                        d10_two_s_temp.append(s)
                        d10_two_alpha_temp.append(valid_alpha)
                        d10_two_label.append(2)
                        d10_two_x.append(x)
                        d10_two_y.append(y)



# two_x =  (two_d_temp[0]-two_s_temp[0])/2
# six_x =  (six_d_temp[0]-six_s_temp[0])/2

# two_y = ((two_d_temp[0]-two_s_temp[0]))/(np.tan(two_alpha_temp[0]))
# six_y = ((six_d_temp[0]-six_s_temp[0]))/(np.tan(six_alpha_temp[0]))



# print(f'length of two alpha {len(two_alpha_temp)}')
# print(f'length of six alpha {len(six_alpha_temp)}')

# combined_alpha = np.array(two_alpha_temp + six_alpha_temp)
# print(f'length of combined alpha {len(combined_alpha)}')
# combined_s = np.array(two_s_temp + six_s_temp)
# combined_d = np.array(two_d_temp + six_d_temp)
# combined_labels = six_label + two_label

d9_combined_alpha = np.array(d9_two_alpha_temp + d9_six_alpha_temp)
d9_combined_s = np.array(d9_two_s_temp + d9_six_s_temp)
d9_combined_d = np.array(d9_two_d_temp + d9_six_d_temp)

d10_combined_alpha = np.array(d10_two_alpha_temp + d10_six_alpha_temp)
d10_combined_s = np.array(d10_two_s_temp + d10_six_s_temp)
d10_combined_d = np.array(d10_two_d_temp + d10_six_d_temp)

# combined_s[combined_d == 4]

# plt.scatter(combined_s[combined_d == 4],combined_alpha[combined_d == 4],color='red')
# plt.plot(six_s_temp[combined_d == 4],six_alpha_temp[combined_d == 4],color='blue',alpha=0.3)
# plt.plot(two_s_temp[combined_d == 4],two_alpha_temp[combined_d == 4],color='green',alpha=0.3)
# plt.show()
# combined_s = np.array(combined_s)
# combined_alpha = np.array(combined_alpha)
# combined_d = np.array(combined_d)

# print(np.shape(combined_s))
# print(np.shape(combined_alpha))
# print(np.shape(combined_s))


# plt.scatter(combined_s[combined_d == 9],combined_alpha[combined_d == 9],color='red')
# plt.plot(six_s_temp[combined_d == 9],six_alpha_temp[combined_d == 9],color='blue',alpha=0.3)
# plt.plot(two_s_temp[combined_d == 9],two_alpha_temp[combined_d == 9],color='green',alpha=0.3)

# plt.scatter(combined_s[combined_d == 10],combined_alpha[combined_d == 10],color='black')
# plt.plot(six_s_temp[combined_d == 10],six_alpha_temp[combined_d == 10],color='blue',alpha=0.3)
# plt.plot(two_s_temp[combined_d == 10],two_alpha_temp[combined_d == 10],color='green',alpha=0.3)
# plt.show()
plt.scatter(d9_combined_s,d9_combined_alpha,color='red',label='d=9')
plt.plot(d9_two_s_temp,d9_two_alpha_temp,color='blue',alpha=0.5,label='Pos 1')
plt.plot(d9_six_s_temp,d9_six_alpha_temp,color='green',alpha=0.5,label='Pos 2')

# plt.scatter(d10_combined_s,d10_combined_alpha,color='black',label='d=10')
# plt.plot(d10_two_s_temp,d10_two_alpha_temp,color='blue',alpha=0.5,ls='--',label='Pos 1')
# plt.plot(d10_six_s_temp,d10_six_alpha_temp,color='green',alpha=0.5,ls='--',label='Pos 2')
plt.legend(loc='upper right')
plt.xlabel("s position (arb.)")
plt.ylabel("Alpha value (rad.)")
plt.show()



# combined_x = []
# combined_y = []
# for i in range(0,len(combined_s)):
#     x_guess = (combined_d[i]-combined_s[i])/2
#     y_guess = ((combined_d[i]-combined_s[i]))/(np.tan(combined_alpha[0]))
#     # x_guess = 5
#     # y_guess = 5 
#     result = spo.basinhopping(func=scatter_difference, niter=100, x0=[x_guess,y_guess], T=0, minimizer_kwargs = {"args":(combined_alpha[i],combined_d[i],combined_s[i]),"method":'BFGS',"bounds":([0,20],[0, 20])})
    
#     if result.x[0] < 0:
#         combined_x.append(0)
#     else:
#         combined_x.append(result.x[0])
#     if result.x[1] < 0:
#         combined_y.append(0)
#     else:
#         combined_y.append(result.x[1])

#     print("Iteration ",i," of ", len(combined_s))


# d = {'combined_s':combined_s,'combined_d':combined_d,'combined_alpha':combined_alpha,'x':combined_x,'y':combined_y,'labels':combined_labels}
# dataframe = pd.DataFrame(data = d)
# dataframe = dataframe.sample(frac=1).reset_index(drop=True)
# y_train = dataframe.labels
# X_train = dataframe.drop(columns='labels')

# plt.scatter(combined_x,combined_y)
# plt.show()

# """
# Old UMAP
# """
# # trans = umap.UMAP(n_neighbors=100, random_state=30, n_components=2, min_dist=1).fit(X_train)
# # # umap.plot.points(trans,labels=y_train)
# # # umap.plot.plt.grid()
# # # umap.plot.plt.show()
# # fig,ax = plt.subplots(1,2)
# # ax[0].scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s= 20, c=y_train, cmap='Spectral')
# # x_ = trans.embedding_[:, 0]
# # y_ = trans.embedding_[:, 1]
# # # plt.title('Embedding of the training set by UMAP', fontsize=24)
# # # plt.grid()
# # ax[1].scatter(x_[x_<5],y_[x_<5],color='black',marker='x',s=10)
# # ax[1].scatter(x_[x_>5],y_[x_>5],color='red',marker='o',s=10)

# # plt.show()



# """
# Clustering UMAP
# """
# standard_embedding = umap.UMAP(
#     n_neighbors=50,
#     min_dist=1,
#     n_components=2,
#     random_state=42,
# ).fit_transform(X_train)

# clusterable_embedding = umap.UMAP(
#     n_neighbors=50,
#     min_dist=1,
#     n_components=2,
#     random_state=42,
# ).fit_transform(X_train)

# plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1],
#             c=y_train, s=20, cmap='Spectral')
# plt.show()

# labels = hdbscan.HDBSCAN(
#     min_samples=10,
#     min_cluster_size=5,
# ).fit_predict(clusterable_embedding)

# print("AHH",labels)
# clustered = (labels >= 0)

# plt.scatter(standard_embedding[~clustered, 0],
#             standard_embedding[~clustered, 1],
#             color=(0.5, 0.5, 0.5),
#             s=20,
#             alpha=0.5)

# plt.scatter(standard_embedding[clustered, 0],
#             standard_embedding[clustered, 1],
#             c=labels[clustered],
#             s=20,
#             cmap='Spectral')

# import itertools
# def indices(lst, item):
#     return [i for i, x in enumerate(lst) if x == item] 

# index_0 = []
# preds_0 = indices(labels,0)
# index_0.append(preds_0)
# merged_0 = list(itertools.chain(*index_0))
# new_index_0=np.sort(list(set(merged_0)))
# cluster_0=X_train.drop(new_index_0,axis=0)

# index_1 = []
# preds_1 = indices(labels,1)
# index_1.append(preds_1)
# merged_1 = list(itertools.chain(*index_1))
# new_index_1=np.sort(list(set(merged_1)))
# cluster_1=X_train.drop(new_index_1,axis=0)

# cluster_0 = cluster_0.drop(columns=['x', 'y'])
# cluster_1 = cluster_1.drop(columns=['x', 'y'])

# loss_on_cluster_0 = loss_minimizer(alpha=cluster_0['combined_alpha'],d=cluster_0['combined_d'],s=cluster_0['combined_s'])
# loss_on_cluster_1 = loss_minimizer(alpha=cluster_1['combined_alpha'],d=cluster_1['combined_d'],s=cluster_1['combined_s'])

# print("Cluster 0:",loss_on_cluster_0)
# print("Cluster 1:",loss_on_cluster_1)

# plt.show()


# # fig = plt.figure()
# # ax = plt.axes(projection='3d')
# # ax.scatter3D(trans.embedding_[:, 0], trans.embedding_[:, 1], trans.embedding_[:, 2],s= 10, c=y_train)

# # plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s= 12, c=y_train, cmap='Spectral')
# # plt.title('Embedding of the training set by UMAP', fontsize=24)
# # plt.grid()
# # plt.show()



# # X_guess = (valid_geometry[0][1]+valid_geometry[0][2])/2
# # X_guess = valid_geometry[0][1]-valid_geometry[0][2]
# # X_guess = 8

# # # Y_guess = ((valid_geometry[0][1]-valid_geometry[0][2]))/(np.tan(valid_geometry[0][0]))
# # Y_guess =  3

# # print("X Guess ", X_guess)
# # print("Y Guess ", Y_guess)

# # res_x = []
# # res_y = []

# # x_err = []
# # y_err = []


# # for i in range(0,5):
# #     result = spo.basinhopping(func=loss_function, x0=[X_guess,Y_guess], niter=1000, T=0, minimizer_kwargs = {"args":(valid_geometry[0],valid_geometry[1],valid_geometry[2]),"bounds":([1,10],[5, 11])})
# #     # result = spo.basinhopping(func=loss_function, x0=[X_guess,Y_guess], niter=400, T=0, minimizer_kwargs = {"args":(valid_geometry[0],valid_geometry[1],valid_geometry[2])})

# #     inv_hessian = result.lowest_optimization_result.hess_inv.todense()
# #     # inv_hessian = result.lowest_optimization_result.hess_inv  

# #     print(inv_hessian)
# #     det_inv_hessian = inv_hessian[0][0] * inv_hessian[1][1] - inv_hessian[0][1] * inv_hessian[1][0]
# #     res_x.append(result.x[0])
# #     res_y.append(result.x[1])
# #     x_err.append(np.sqrt(inv_hessian[1][1]/det_inv_hessian))
# #     y_err.append(np.sqrt(inv_hessian[0][0]/det_inv_hessian))

# # res_x =  np.array(res_x)
# # res_y = np.array(res_y)
# # print("",np.mean(res_x)," +/- ",np.mean(x_err))
# # print("",np.mean(res_y)," +/- ",np.mean(y_err))











