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



 

X_bounds = [1,10]
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
                if (x == 9) and (y==4):
                    valid_alpha = alpha_calc(x,y,d,s)
                    valid_geometry.append([x,y,d,s,valid_alpha])
                    six_d_temp.append(d)
                    six_s_temp.append(s)
                    six_alpha_temp.append(valid_alpha)
                    six_label.append(6)
                    six_x.append(x)
                    six_y.append(y)

                if (x == 2) and (y==7):
                    valid_alpha = alpha_calc(x,y,d,s)
                    valid_geometry.append([x,y,d,s,valid_alpha])
                    two_d_temp.append(d)
                    two_s_temp.append(s)
                    two_alpha_temp.append(valid_alpha)
                    two_label.append(2)
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
for i in range(0,len(combined_s)):
    x_guess = (combined_d[i]-combined_s[i])/2
    y_guess = ((combined_d[i]-combined_s[i]))/(np.tan(combined_alpha[0]))
    result = spo.basinhopping(func=scatter_difference, niter=100, x0=[x_guess,y_guess], T=0, minimizer_kwargs = {"args":(combined_alpha[i],combined_d[i],combined_s[i]),"method":'BFGS',"bounds":([0,20],[0, 20])})
    
    if result.x[0] < 0:
        combined_x.append(0)
    else:
        combined_x.append(result.x[0])
    if result.x[1] < 0:
        combined_y.append(0)
    else:
        combined_y.append(result.x[1])

    print("Iteration ",i," of ", len(combined_s))


d = {'combined_s':combined_s,'combined_d':combined_d,'combined_alpha':combined_alpha,'x':combined_x,'y':combined_y,'labels':combined_labels}
dataframe = pd.DataFrame(data = d)
dataframe = dataframe.sample(frac=1).reset_index(drop=True)
y_train = dataframe.labels
X_train = dataframe.drop(columns='labels')

plt.scatter(combined_x,combined_y)
plt.show()

"""
Old UMAP
"""
# trans = umap.UMAP(n_neighbors=100, random_state=30, n_components=2, min_dist=1).fit(X_train)
# # umap.plot.points(trans,labels=y_train)
# # umap.plot.plt.grid()
# # umap.plot.plt.show()
# fig,ax = plt.subplots(1,2)
# ax[0].scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s= 20, c=y_train, cmap='Spectral')
# x_ = trans.embedding_[:, 0]
# y_ = trans.embedding_[:, 1]
# # plt.title('Embedding of the training set by UMAP', fontsize=24)
# # plt.grid()
# ax[1].scatter(x_[x_<5],y_[x_<5],color='black',marker='x',s=10)
# ax[1].scatter(x_[x_>5],y_[x_>5],color='red',marker='o',s=10)

# plt.show()



"""
Clusters
"""
standard_embedding = umap.UMAP(
    n_neighbors=50,
    min_dist=1,
    n_components=2,
    random_state=42,
).fit_transform(X_train)

clusterable_embedding = umap.UMAP(
    n_neighbors=50,
    min_dist=1,
    n_components=2,
    random_state=42,
).fit_transform(X_train)

plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1],
            c=y_train, s=20, cmap='Spectral')
plt.show()

labels = hdbscan.HDBSCAN(
    min_samples=10,
    min_cluster_size=5,
).fit_predict(clusterable_embedding)

print("AHH",labels)
clustered = (labels >= 0)

plt.scatter(standard_embedding[~clustered, 0],
            standard_embedding[~clustered, 1],
            color=(0.5, 0.5, 0.5),
            s=20,
            alpha=0.5)

plt.scatter(standard_embedding[clustered, 0],
            standard_embedding[clustered, 1],
            c=labels[clustered],
            s=20,
            cmap='Spectral')

plt.show()





# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(trans.embedding_[:, 0], trans.embedding_[:, 1], trans.embedding_[:, 2],s= 10, c=y_train)

# plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s= 12, c=y_train, cmap='Spectral')
# plt.title('Embedding of the training set by UMAP', fontsize=24)
# plt.grid()
# plt.show()



# X_guess = (valid_geometry[0][1]+valid_geometry[0][2])/2
# X_guess = valid_geometry[0][1]-valid_geometry[0][2]
# X_guess = 8

# # Y_guess = ((valid_geometry[0][1]-valid_geometry[0][2]))/(np.tan(valid_geometry[0][0]))
# Y_guess =  3

# print("X Guess ", X_guess)
# print("Y Guess ", Y_guess)

# res_x = []
# res_y = []

# x_err = []
# y_err = []


# for i in range(0,5):
#     result = spo.basinhopping(func=loss_function, x0=[X_guess,Y_guess], niter=1000, T=0, minimizer_kwargs = {"args":(valid_geometry[0],valid_geometry[1],valid_geometry[2]),"bounds":([1,10],[5, 11])})
#     # result = spo.basinhopping(func=loss_function, x0=[X_guess,Y_guess], niter=400, T=0, minimizer_kwargs = {"args":(valid_geometry[0],valid_geometry[1],valid_geometry[2])})

#     inv_hessian = result.lowest_optimization_result.hess_inv.todense()
#     # inv_hessian = result.lowest_optimization_result.hess_inv  

#     print(inv_hessian)
#     det_inv_hessian = inv_hessian[0][0] * inv_hessian[1][1] - inv_hessian[0][1] * inv_hessian[1][0]
#     res_x.append(result.x[0])
#     res_y.append(result.x[1])
#     x_err.append(np.sqrt(inv_hessian[1][1]/det_inv_hessian))
#     y_err.append(np.sqrt(inv_hessian[0][0]/det_inv_hessian))

# res_x =  np.array(res_x)
# res_y = np.array(res_y)
# print("",np.mean(res_x)," +/- ",np.mean(x_err))
# print("",np.mean(res_y)," +/- ",np.mean(y_err))