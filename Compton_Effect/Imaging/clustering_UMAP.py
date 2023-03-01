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

    X_guess = 2
    Y_guess = 15

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
x_2_true = 11
y_1_true = 5
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

combined_x = np.array(combined_x)
combined_d = np.array(combined_d)
combined_s = np.array(combined_s)
combined_alpha = np.array(combined_alpha)
combined_y = np.array(combined_y)

combined_x = combined_x[combined_y > 1.5]
combined_d = combined_d[combined_y > 1.5]
combined_s = combined_s[combined_y > 1.5]
combined_alpha = combined_alpha[combined_y > 1.5]

new_labels = [combined_labels[i] for i  in range(0,len(combined_labels)) if combined_y[i]>1.5]
combined_y = combined_y[combined_y > 1.5]

combined_labels = new_labels



d = {'combined_s':combined_s,'combined_d':combined_d,'combined_alpha':combined_alpha,'x':combined_x,'y':combined_y,'labels':combined_labels}
dataframe = pd.DataFrame(data = d)
dataframe = dataframe.sample(frac=1).reset_index(drop=False)
y_train = dataframe.labels
X_train = dataframe.drop(columns='labels')
X_train = X_train.drop(columns='index')


print("COlumns!! ",X_train.columns)
print("Dataframe",X_train)

plt.scatter(combined_x,combined_y)
plt.show()


"""
Clustering UMAP
"""
# X_train = X_train.drop(columns=['x','y'])
# standard_embedding = umap.UMAP(
#     n_neighbors=50,
#     min_dist=1,
#     n_components=2,
#     random_state=42,
# ).fit_transform(X_train)

X_train_norm = (X_train-X_train.mean())/X_train.std()
clusterable_embedding = umap.UMAP(
    n_neighbors=5,
    min_dist=0,
    n_components=2,
    random_state=42,
).fit_transform(X_train_norm)

# print("COlumns!! ",X_train.columns)
plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1],
            c=y_train, s=20, cmap='Spectral')
plt.show()

# labels = hdbscan.HDBSCAN(
#     min_samples=10,
#     min_cluster_size=70,
# ).fit_predict(clusterable_embedding)

model = GaussianMixture(n_components=2,n_init=4,max_iter=500)
# fit the model
model.fit(clusterable_embedding)
# assign a cluster to each example
yhat = model.predict(clusterable_embedding)
# retrieve unique clusters
clusters = np.unique(yhat)
# create scatter plot for samples from each cluster
coordinates = []
preds = []
print("cluster embed",clusterable_embedding)
print("cluster shape",np.shape(clusterable_embedding))
for i,cluster in enumerate(clusters):
 # get row indexes for samples with this cluster
 row_ix = np.where(yhat == cluster)
 # create scatter of these samples
 plt.scatter(clusterable_embedding[row_ix, 0], clusterable_embedding[row_ix, 1],label="Cluster "+str(i))
 coordinates.append([clusterable_embedding[row_ix, 0], clusterable_embedding[row_ix, 1]])
 preds.append(row_ix)
# show the plot
print("All coords",coordinates)
print("Cluster 0",coordinates[0])
print("Cluster 1",coordinates[1])


plt.xlabel("X position (arb.)")
plt.ylabel("Y position (arb.)")
plt.legend(loc="upper right")
plt.show()

print(f'preds {list(preds)}')
print(f'preds[0][0] {list(preds)[0][0]}')
print(f'preds[1] {list(preds)[1]}')
print(f'preds[1][0] {list(preds)[1][0]}')
print(f'preds shape {np.shape(preds)}')
marker = []
for i in range(len(X_train)):
    if i in preds[0][0]:
        marker.append(0)
    
    elif i in preds[1][0]:
        marker.append(1)

X_train['marker'] = marker

cluster_0 = X_train[X_train['marker'] == 0]
cluster_1 = X_train[X_train['marker'] == 1]



# cluster_0 = np.array(coordinates[0])
# cluster_1 = np.array(coordinates[0])


# cluster_0 = cluster_0.drop(columns=['x', 'y'])
# cluster_1 = cluster_1.drop(columns=['x', 'y'])

loss_on_cluster_0 = loss_minimizer(alpha=cluster_0['combined_alpha'],d=cluster_0['combined_d'],s=cluster_0['combined_s'])
loss_on_cluster_1 = loss_minimizer(alpha=cluster_1['combined_alpha'],d=cluster_1['combined_d'],s=cluster_1['combined_s'])


print("*************************************")
print("Results")
print("*************************************")
print("")
print("Cluster 0:",loss_on_cluster_0)
print("Cluster 1:",loss_on_cluster_1)
print("")
print(f"True Cluster 0: [{x_1_true}, {y_1_true}]")
print(f"True Cluster 1: [{x_2_true}, {y_2_true}]")
print("")
print("The clusters may be swapped.")

# plt.show()


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
