import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xraydb

R_0 = 10973731.6
h = 6.63e-34
c = 3e8
# elements = ['H', 'He', 'Li', 'Be','B','C','N','O','F','N','Na','Mg','Al','Si','Ph','S',"Cl","Ar","K","Ca"]

ka1 = []
kb1 = []
for i in range(0,len(elements)):
    for name, line in xraydb.xray_lines(elements[i], 'K').items():
        if name == 'Ka1':
            # ka1 = line.energy
            ka1.append(line.energy)
        elif name == 'Kb1':
            kb1.append(line.energy)
print(f"K_alpha energies: {ka1}")
print(f"K_beta energies: {kb1}")

