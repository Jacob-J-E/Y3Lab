import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("ATLAS")


data = pd.read_csv(r"X-Ray\Data\16-01-2022\NaCl Full Data.csv",skiprows=0)
print(data)


angle = data['angle']
wav = data['wav / pm']
energy = data['E / keV']
count_0 = data['R_0 / 1/s']

A = 564.02e-12
ENERGY1 = 17.443e3*1.6e-19
ENERGY2 = 19.651e3*1.6e-19

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
for h in range_max:
    for k in range_max:
        for l in range_max:
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
        plt.axvline(x, color = 'g')
    else:
        plt.axvline(x, color = 'g', label = 'even fcc lattice')

plt.scatter(angle,count_0, label = 'Data')


plt.legend()
plt.show()

