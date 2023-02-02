import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import scipy.optimize as spo
hep.style.use("ATLAS")

def exp(x,A,m,c):
    return A * np.exp(-1 * m * x) + c


data = pd.read_csv(r"X-Ray\Data\16-01-2023\NaCl Full Data.csv",skiprows=0)
print(data)


angle = data['angle']
wav = data['wav / pm']
energy = data['E / keV']
count_0 = data['R_0 / 1/s']

A = 564.02e-12
# ENERGY1 = 17.443e3*1.6e-19
# ENERGY2 = 19.651e3*1.6e-19
ENERGY1 = 8046.3*1.6e-19
ENERGY2 = 8903.9*1.6e-19

def calculate_angle(h,k,l,energy):
    wavelength = (6.63e-34 * 3e8)/energy
    sin_angle = np.sqrt((wavelength**2/(4*A**2))*(h**2+k**2+l**2))
    angle = np.arcsin(sin_angle)*180/np.pi
    return angle

primitive = set()
bcc = set()
fcc_even = set()
fcc_odd = set()
hcp = set()

def bcc_check(h,k,l):
    if (h+k+l)%2 == 0:
        return True
    return False

def fcc_check_even(h,k,l):
    if (h%2 == 0 and k%2 == 0 and l%2 == 0):
        return True
    return False

def fcc_check_odd(h,k,l):
    if (h%2 == 1 and k%2 == 1 and l%2 == 1):
        return True
    return False

def hcp_check(h,k,l):
    if l%2 == 1 and h+2*k == 3*(h+k+l):
        return False
    return True

range_max = range(0,10)
range_one = range(0,10)
for h in range_max:
    for k in range_one:
        for l in range_one:
            angle1 = calculate_angle(h,k,l,ENERGY1)
            angle2 = calculate_angle(h,k,l,ENERGY2)
            primitive.add(angle1)
            primitive.add(angle2)
            if bcc_check(h,k,l):
                bcc.add(angle1)
                bcc.add(angle2)
            if fcc_check_even(h,k,l):
                fcc_even.add(angle1)
                fcc_even.add(angle2)
            if fcc_check_odd(h,k,l):
                fcc_odd.add(angle1)
                fcc_odd.add(angle2)    
            if hcp_check(h,k,l):
                hcp.add(angle1)
                hcp.add(angle2)

angle_upper_threshold = 30

primitive = [i for i in primitive if i <= angle_upper_threshold]
bcc = [i for i in bcc if i <= angle_upper_threshold]
fcc_even = [i for i in fcc_even if i <= angle_upper_threshold]
fcc_odd = [i for i in fcc_odd if i <= angle_upper_threshold]
hcp = [i for i in hcp if i <= angle_upper_threshold]

# print(len(primitive))
# print(len(bcc))

# for idx,x in enumerate(primitive):
#     if not(idx == len(primitive)-1):
#         plt.axvline(x, color = 'r')
#     else:
#         plt.axvline(x, color = 'r', label = 'primitive lattice')

# for idx,x in enumerate(bcc):
#     if not(idx == len(bcc)-1):
#         plt.axvline(x, color = 'b')
#     else:
#         plt.axvline(x, color = 'b', label = 'bcc lattice')

for idx,x in enumerate(fcc_even):
    if not(idx == len(fcc_even)-1):
        plt.axvline(x, color = 'black')
    #else:
        plt.axvline(x, color = 'black')
#plt.plot([0,0],[0,0],color='black',label = 'Even FCC Lattice')


# angle_line = np.concatenate((angle[(angle > 7.9) & (angle < 12.1)],angle[(angle > 4.4) & (angle < 6.1)]))
# count_line = np.concatenate((count_0[(angle > 7.9) & (angle < 12.1)],count_0[(angle > 4.4) & (angle < 6.1)]))
# lambda_guess = len(angle_line) / sum(angle_line)
# line_guess = [max(count_line),lambda_guess,max(count_line)]
# line_params, line_cov = spo.curve_fit(exp,angle_line,count_line,line_guess)


plt.plot(angle,count_0, label = 'Experimental Data')
plt.xlabel("Angle (degrees)")
plt.ylabel("Count rate /s")
#plt.plot(angle,exp(angle,*line_params),color='red',label="Exponential Background Fit-Line")
#plt.plot(angle,count_0-exp(angle,*line_params),color='green',label="Background-Reduced Data")
plt.legend()
plt.grid()
plt.show()

# plt.plot(np.sort(energy),count_0)
# plt.show()

# plt.plot(angle,count_0,color='blue',label="Experimental Data")
# plt.plot(angle,exp(angle,*line_params),color='red',label="Exponential Background Fit-Line")
# plt.plot(angle,count_0-exp(angle,*line_params),color='black',label="Background-Reduced Data")
# plt.grid()
# plt.show()



# fig, ax = plt.subplots(figsize=[8,5])

# ax.plot(angle,count_0,color='blue',label="Experimental Data")

# axins = zoomed_ins