import mendeleev as me
import numpy as np
import matplotlib.pyplot as plt

#sconst
# Z = []
# Z_eff = []
# for i in range(1,100):
#     element = me.element(i)
#     Z.append(i)
#     print(element.sconst)
#     Z_eff.append(element.sconst)

# plt.scatter(Z,Z_eff)
# plt.show()


ele = me.element('Li')
print(ele.sconst)