import numpy as np
import scipy.optimize as spo
from alive_progress import alive_bar

class Generating_data:
    def __init__(self,x_1_true,x_2_true,y_1_true,y_2_true, X_bounds = [1,20],Y_bounds = [1,10], method = "BFGS") -> None:
        self.method = method
        self.x_1_true = x_1_true
        self.x_2_true = x_2_true
        self.y_1_true = y_1_true
        self.y_2_true = y_2_true
        self.X_bounds = X_bounds
        self.Y_bounds = Y_bounds
    
    def alpha_calc(self,X,Y,d,s):
        theta = np.arctan2(Y,(X-s))
        phi = np.arctan2(Y,(d-X))
        alpha = np.pi - theta - phi
        return alpha
    
    def scatter_difference(self,coordinates: list, alpha: np.array, d:np.array, s:np.array):
        X, Y = coordinates
        theta = np.arctan2(Y,(X-s))
        phi = np.arctan2(Y,(d-X))
        inner_value = np.pi - alpha - theta - phi
        return np.abs(inner_value)**2

    def generate_basinhopping_x_y(self):
        # Declare true geometry
        x_1_true = self.x_1_true
        x_2_true = self.x_2_true
        y_1_true = self.y_1_true
        y_2_true = self.y_2_true

        X_bounds = self.X_bounds
        Y_bounds = self.Y_bounds

        one_alpha_temp = []
        one_s_temp = []
        one_d_temp = []
        two_alpha_temp = []
        two_s_temp = []
        two_d_temp = []
        for x in range(X_bounds[0],X_bounds[1]+1):
            for y in range(Y_bounds[0],Y_bounds[1]+1):
                for s in range(X_bounds[0],X_bounds[1]+1):
                    for d in range(X_bounds[0]+1, X_bounds[1]):
                        if (x == x_1_true) and (y==y_1_true):
                            valid_alpha = self.alpha_calc(x,y,d,s)
                            one_d_temp.append(d)
                            one_s_temp.append(s)
                            one_alpha_temp.append(valid_alpha)
                        if (x == x_2_true) and (y==y_2_true):
                            valid_alpha = self.alpha_calc(x,y,d,s)
                            two_d_temp.append(d)
                            two_s_temp.append(s)
                            two_alpha_temp.append(valid_alpha)

        combined_alpha = np.array(two_alpha_temp + one_alpha_temp)
        combined_s = np.array(two_s_temp + one_s_temp)
        combined_d = np.array(two_d_temp + one_d_temp)
        combined_x = []
        combined_y = []
        print('Generating Data...')
        with alive_bar(len(combined_s)) as bar:
            for i in range(0,len(combined_s)):

                x_guess = float((combined_d[i]+combined_s[i])/2)
                y_guess = np.abs(0.5*((combined_d[i]-combined_s[i]))/(np.tan(combined_alpha[0])))

                result = spo.basinhopping(func=self.scatter_difference, niter=40, x0=list([x_guess,y_guess]), T=0, minimizer_kwargs = {"args":(combined_alpha[i],combined_d[i],combined_s[i]),"method":self.method,"bounds":((X_bounds[0], Y_bounds[0]),(X_bounds[1], Y_bounds[1]))})

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

        return combined_x,combined_y