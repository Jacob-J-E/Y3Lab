import numpy as np
import matplotlib.pyplot as plt

h = 6.63e-34
m = 9.11e-31
c = 3e8
r_0 = 2.82e-15

def Klein_Nishina(x,freq):
    gamma = h * freq / (m * c**2)
    cross_sec = r_0**2 * 0.5 * (1+np.cos(x)**2)\
        *(1/(1+gamma*(1-np.cos(x)))**2)\
            *(1+(gamma**2*(1-np.cos(x))**2)/((1+np.cos(x)**2)*(1+gamma*(1-np.cos(x)))))
    return cross_sec


x = np.arange(0,90,0.1)

y = Klein_Nishina(x,4.226e22)

plt.plot(x,y)
plt.show()