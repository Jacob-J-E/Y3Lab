import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter

res_array = np.loadtxt("res_array.csv", delimiter=",")
c_space = np.loadtxt("c_space.csv", delimiter=",")

plt.scatter(c_space,res_array)

print(len(c_space), 'Length of C space initially')
print(len(res_array), 'Length of res array initially')


inital_res = res_array[0]
if max(res_array[(c_space > c_space[0] + 2)]) >= inital_res:
    temp_res_array = []
    temp_res_array.append(inital_res)
    print(inital_res)
    triggered = False 

    for i in range(1,len(res_array)):
        if res_array[i] >= inital_res or triggered == True:
            triggered = True
            temp_res_array.append(0)
        else:
            temp_res_array.append(res_array[i])
    print(len(c_space))
    print(len(temp_res_array))
    res_array = np.array(temp_res_array)

    c_space = c_space[(res_array > 0)]
    res_array = res_array[(res_array > 0)]

    print(len(c_space), 'Length of C space after high fluctuations')
    print(len(res_array), 'Length of res array after high fluctuations')

plt.scatter(c_space,res_array)

sav_gol_num = 87
res_array_range_bounds = res_array[(c_space[0]+1 < c_space) & (c_space[-1]-1 > c_space)]
c_space_range_bounds = c_space[(c_space[0]+1 < c_space) & (c_space[-1]-1 > c_space)]
savgol_range_bounds = savgol_filter(res_array_range_bounds,sav_gol_num,3)
savgol = savgol_filter(res_array,sav_gol_num,3)

plt.plot(c_space_range_bounds,savgol_range_bounds)

index_less = argrelextrema(savgol_range_bounds, np.less)
c_min = c_space[index_less]
savgol_min = savgol[index_less]

index_greater = argrelextrema(savgol_range_bounds, np.greater)
c_greater = c_space[index_greater]
savgol_greater = savgol[index_greater]
print(len(savgol_min), 'Length of savgol minimum')
print(len(savgol_greater), 'Length of savgol greater')

print(savgol_greater[0], 'first element greater savgol')
print(c_greater[0], 'first element greater c space')
print(savgol_min[0], 'first element minimum savgol')
print(c_min[0], 'first element greater c space')
ranges = []
if len(savgol_min) > len(savgol_greater) :
    for i in range(len(savgol_min)-1):
        if (savgol_greater[i] - savgol_min[i]) not in ranges: 
            ranges.append((savgol_greater[i] - savgol_min[i]))
        if (savgol_greater[i] - savgol_min[i+1]) not in ranges: 
            ranges.append((savgol_greater[i] - savgol_min[i+1]))
elif len(savgol_min) == len(savgol_greater) :
    for i in range(len(savgol_greater)):
        if (savgol_greater[i] - savgol_min[i]) not in ranges: 
            ranges.append((savgol_greater[i] - savgol_min[i]))
elif len(savgol_min) < len(savgol_greater) :
    for i in range(len(savgol_greater)-1):
        if (savgol_greater[i] - savgol_min[i]) not in ranges: 
            ranges.append((savgol_greater[i] - savgol_min[i]))
        if (savgol_greater[i+1] - savgol_min[i]) not in ranges: 
            ranges.append((savgol_greater[i+1] - savgol_min[i]))

print(ranges)
ranges = np.array(ranges)
q_3 = np.percentile(ranges, 75)
q_1 = np.percentile(ranges, 25)
iqr = q_3 - q_1
print(iqr)
ranges = ranges[(q_3+8*iqr > ranges) & (q_1-8*iqr < ranges)]
high_percentile = np.percentile(ranges, 99) 
print(high_percentile, 'high_percentile')
high_percentile = q_3+4*iqr
print(high_percentile, 'high_percentile')

index_less = argrelextrema(savgol, np.less)
c_min = c_space[index_less]
savgol_min = savgol[index_less]

index_greater = argrelextrema(savgol, np.greater)
c_greater = c_space[index_greater]
savgol_greater = savgol[index_greater]

# biggest_range = max(savgol_greater) - min(savgol_min)
# smallest_range = min(savgol_greater) - max(savgol_min)
# mid = (biggest_range + smallest_range)/2
# print(mid, 'mid')

if savgol_greater[0] > savgol_min[0]:
    c_greater =  np.array([c_space[0]] + list(c_greater))
    savgol_greater =  np.array([savgol[0]] + list(savgol_greater))


mini = []

max_index = min(len(savgol_min),len(savgol_greater))

prev_savgol_min = savgol_min[0]
prev_savgol_greater_left = savgol_greater[0]
prev_savgol_greater_right = savgol_greater[1]


count = 0
for i in range(len(savgol_min[:11])):

    if (abs(savgol_greater[i] - savgol_min[i]) > 0.01):
        prev_savgol_greater_left = savgol_greater[i]
    if (abs(savgol_greater[i+1] - savgol_min[i]) > 0.01):
        prev_savgol_greater_right = savgol_greater[i+1]

    # print(f'savgol_greater {savgol_greater[i]} and savgol_min {savgol_min[i]} difference {savgol_greater[i] - savgol_min[i]}')
    # print(f'prev_savgol_greater_left {prev_savgol_greater_left} and savgol_min {savgol_min[i]} difference {prev_savgol_greater_left - savgol_min[i]}')
    # print(f'prev_savgol_greater_right {prev_savgol_greater_right} and savgol_min {savgol_min[i]} difference {prev_savgol_greater_right - savgol_min[i]}')
    # print(i < len(savgol_greater)-1)
    # print(prev_savgol_greater_left - savgol_min[i] > high_percentile)
    # print(prev_savgol_greater_right - savgol_min[i] > high_percentile)
    if count == 2:
        break
    #if (i < len(savgol_greater)-1 and savgol_greater[i] - savgol_min[i] > high_percentile and savgol_greater[i+1] - savgol_min[i] > high_percentile):
    if (i < len(savgol_greater)-1 and prev_savgol_greater_left - savgol_min[i] > high_percentile and prev_savgol_greater_right - savgol_min[i] > high_percentile):
                    if savgol_min[i] not in mini:
                            mini.append(savgol_min[i])
                            print(savgol_min[i])
                            print(c_min[i])
                            count += 1

# if len(savgol_min) > len(savgol_greater):
#     for i in range(len(savgol_min)-1):
#         if count == 2:
#             break
#         if savgol_greater[i] - savgol_min[i] > high_percentile:
#             if i == 0 or savgol_greater[i-1] - savgol_min[i] > high_percentile:
#                 if savgol_min[i] not in mini:
#                     mini.append(savgol_min[i])
#                     print(savgol_min[i])
#                     print(c_min[i])
#                     count += 1
#         if count == 2:
#             break
#         if savgol_greater[i] - savgol_min[i+1] > high_percentile:
#             if savgol_min[i+1] not in mini:
#                 mini.append(savgol_min[i+1])
#                 print(savgol_min[i+1])
#                 print(c_min[i+1])
#                 count += 1 
# elif len(savgol_min) == len(savgol_greater) :
#     for i in range(len(savgol_min)):
#         if count == 2:
#             break
#         if savgol_greater[i] - savgol_min[i] > high_percentile:
#             if savgol_min[i] not in mini:
#                 mini.append(savgol_min[i])
#                 print(savgol_min[i])
#                 print(c_min[i])
#                 count += 1
# elif len(savgol_min) < len(savgol_greater) :
#     for i in range(len(savgol_greater)-1):
#         if count == 2:
#             break
#         if savgol_greater[i] - savgol_min[i] > high_percentile:
#             if savgol_min[i] not in mini:
#                 mini.append(savgol_min[i])
#                 print(savgol_min[i])
#                 print(c_min[i])
#                 count += 1
#         if count == 2:
#             break
#         if savgol_greater[i+1] - savgol_min[i] > high_percentile:
#             if savgol_min[i] not in mini:
#                 mini.append(savgol_min[i])
#                 print(savgol_min[i])
#                 print(c_min[i])
#                 count += 1 

# print(c_greater, ' c_greater')
# print(savgol_greater, ' savgol_greater')
print(len(savgol_greater))
# print(c_min, ' c_min')
# print(savgol_min, ' savgol_min')
print(len(savgol_min))


plt.show()