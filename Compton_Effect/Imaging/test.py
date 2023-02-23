import numpy as np

data = np.array([[1,2],[3,4],[5,6],[7,8]])


print(data[:,0])
print(data[:,1])


data = data[data[:,1] >4]
print(data)