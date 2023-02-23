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
import scipy.optimize as spo
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
x_2_true = 4
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
combined_y = np.array(combined_y)
combined_x = np.array(combined_x)
combined_x = combined_x[combined_y > 1.5]
combined_y = combined_y[combined_y > 1.5]


# data = pd.read_csv('Compton_Effect\Imaging\data_x_y_9y_3y.csv')
# combined_x = np.array(data['x'])
# combined_y = np.array(data['y'])

plt.figure(1)
plt.scatter(combined_x,np.array(combined_y))
# plt.plot([min(combined_x),max(combined_x)],[6.78+bounds,6.78+bounds])
# plt.plot([min(combined_x),max(combined_x)],[6.78-bounds,6.78-bounds])
plt.axvline(np.median(combined_x), color = 'black')
plt.axvline(np.median(combined_x)+3, color = 'black')

med = np.median(combined_x)
mid_shift = med + 3
medium_range_y = combined_y[(combined_x > med - 0.5) & (combined_x < med + 0.5)]
medium_range_y_shift = combined_y[(combined_x > mid_shift - 0.5) & (combined_x < mid_shift + 0.5)]
combine = list(medium_range_y_shift) + list(medium_range_y)
combine_sorted = sorted(combine)
#medium_range_y_sorted = sorted(medium_range_y)
fh_med = combine_sorted[:int(len(combine_sorted)/2)]
sh_med = combine_sorted[int(len(combine_sorted)/2):]




def gaussian(x, a, b, c):
    return (a * np.exp(-((x - b) ** 2) / (2 * c ** 2)))

# first histogram - lower

num_of_bins = 20

hist_fh, bin_edges_fh = np.histogram(fh_med, bins=num_of_bins)

center_of_bins_fh = (bin_edges_fh[:-1] + bin_edges_fh[1:]) / 2



first_index_fh = np.where(hist_fh == max(hist_fh))
first_index_fh = first_index_fh[0][0]

first_mean_guess_fh = center_of_bins_fh[first_index_fh]
width_fh = (center_of_bins_fh[1]- center_of_bins_fh[0])

print(first_mean_guess_fh)
print(width_fh)
print(max(hist_fh))



guess_e1 = [max(hist_fh),first_mean_guess_fh,width_fh]
params_e1, cov_e1 = spo.curve_fit(gaussian,center_of_bins_fh,hist_fh,guess_e1)
print(params_e1)
domain_fh = np.linspace(min(bin_edges_fh),max(bin_edges_fh),num = 10000)

plt.figure(2)
plt.hist(fh_med, bins = num_of_bins)
plt.scatter(center_of_bins_fh,hist_fh, marker = 'x')
plt.plot(domain_fh,gaussian(domain_fh,*params_e1))


hist_sh, bin_edges_sh = np.histogram(sh_med, bins=num_of_bins)

center_of_bins_sh = (bin_edges_sh[:-1] + bin_edges_sh[1:]) / 2



first_index_sh = np.where(hist_sh == max(hist_sh))
first_index_sh = first_index_sh[0][0]

first_mean_guess_sh = center_of_bins_sh[first_index_sh]
width_sh = (center_of_bins_sh[1]- center_of_bins_sh[0])

print(first_mean_guess_sh)
print(width_sh)
print(max(hist_sh))



guess_e2 = [max(hist_sh),first_mean_guess_sh,width_sh]
params_e2, cov_e2 = spo.curve_fit(gaussian,center_of_bins_sh,hist_sh,guess_e2)
print(params_e2)
domain_sh = np.linspace(min(bin_edges_sh),max(bin_edges_sh),num = 10000)

plt.figure(3)
plt.hist(sh_med, bins = num_of_bins)
plt.scatter(center_of_bins_sh,hist_sh, marker = 'x')
plt.plot(domain_sh,gaussian(domain_sh,*params_e2))



# iqr = np.percentile(medium_range_y, 75) - np.percentile(medium_range_y, 25)

# h = 2 * iqr / len(medium_range_y)**(1/2)

# num_of_bins = 50
# #num_of_bins = int((max(medium_range_y) - min(medium_range_y))/h)
# def gaussian(x, a, b, c,e,f,g):
#     return (a * np.exp(-((x - b) ** 2) / (2 * c ** 2)) + e * np.exp(-((x - f) ** 2) / (2 * g ** 2)))

# hist, bin_edges = np.histogram(medium_range_y, bins=num_of_bins)

# center_of_bins = (bin_edges[:-1] + bin_edges[1:]) / 2

# plt.figure(2)
# plt.hist(medium_range_y, bins = num_of_bins)
# plt.scatter(center_of_bins,hist, marker = 'x')
# plt.show()
# #first_mean_guess = center_of_bins[hist[:int(len(center_of_bins)/2)].index(max(hist[:int(len(center_of_bins)/2)]))]
# first_mean_guess = center_of_bins[np.where(hist[:int(len(center_of_bins)/2)] == max(hist[:int(len(center_of_bins)/2)]))][0]
# arr = np.where(hist[int(len(center_of_bins)/2):] == max(hist[int(len(center_of_bins)/2):]))
# print(arr[0][0],'arraay')
# second_mean_guess = center_of_bins[int(len(center_of_bins)/2) - 1 + arr[0][0]]
# #second_mean_guess = center_of_bins[hist[int(len(center_of_bins)/2):].index(max(hist[int(len(center_of_bins)/2):]))]
# print(max(hist[int(len(center_of_bins)/2):]))
# print(max(hist[:int(len(center_of_bins)/2)]))
# print(center_of_bins)
# print(np.where(hist[int(len(center_of_bins)/2):] == max(hist[int(len(center_of_bins)/2):])))
# print(center_of_bins[np.where(hist[int(len(center_of_bins)/2):] == max(hist[int(len(center_of_bins)/2):]))])
# print(first_mean_guess,' first mean')
# print(second_mean_guess,' second mean')
# width = (center_of_bins[1]- center_of_bins[0])/4
# print(width,' width')

# guess_e1 = [max(hist[:int(len(center_of_bins)/2)]),first_mean_guess,width,max(hist[int(len(center_of_bins)/2):]),second_mean_guess,width]
# params_e1, cov_e1 = spo.curve_fit(gaussian,center_of_bins,hist,guess_e1)
# domain = np.linspace(min(bin_edges),max(bin_edges),num = 10000)
# plt.figure(2)
# plt.hist(medium_range_y, bins = num_of_bins)
# plt.scatter(center_of_bins,hist, marker = 'x')
# plt.plot(domain,gaussian(domain,*params_e1))

# plt.show()

combined_x = combined_x[combined_y > params_e1[1]-2]
combined_y = combined_y[combined_y > params_e1[1]-2]

# data = {'x':combined_x,'y':combined_y}
# data = pd.DataFrame(data = data)
# data.to_csv('data_x_y.csv')

# data = pd.read_csv('data_x_y.csv')
# combined_x = np.array(data['x'])
# combined_y = np.array(data['y'])

c_space = np.linspace(0,max(combined_y),num = 10000)

bounds = (params_e2[1] - params_e1[1])/4 #- max(params_e1[2], params_e2[2])
#bounds = max(3*params_e1[2], 3*params_e2[2])

sav_gol_num = int(bounds*4)*2 * 43 + 1

def sum_res(c,y_data):
    y_data = np.array(y_data)
    y_data = y_data[(y_data < c+bounds) & (y_data > c-bounds)]
    res = (c-y_data)**2
    if len(y_data) != 0:
        return sum(res)/len(y_data)
    return sum(res)

res_array = []
for i in c_space:
    res_array.append(sum_res(i,combined_y))


print('savgol_num', sav_gol_num)
plt.figure(4)
plt.plot(c_space,savgol_filter(res_array,sav_gol_num,3), color = 'orange')
plt.scatter(c_space,res_array)
plt.plot(c_space,savgol_filter(res_array,sav_gol_num,3), color = 'orange')


plt.show()