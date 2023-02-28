import matplotlib.pyplot as plt
import numpy as np


from generating_data import Generating_data
from inital_y_guess import Inital_y_guess


data_obj = Generating_data(x_1_true = 12,x_2_true = 4,y_1_true = 5,y_2_true= 3, X_bounds = [1,20],Y_bounds = [1,10])
data_x,data_y = data_obj.generate_basinhopping_x_y()

data_x = data_x[data_y > 1.5]
data_y = data_y[data_y > 1.5]

# inital_y_obj = Inital_y_guess(combined_x = data_x,combined_y = data_y)
# params_fh, cov_fh, params_sh, cov_sh = inital_y_obj.guess_y()
# data_x = data_x[data_y > params_fh[1]-2]
# data_y = data_y[data_y > params_fh[1]-2]

plt.scatter(data_x,data_y)
plt.show()