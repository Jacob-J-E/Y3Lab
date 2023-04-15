import matplotlib.pyplot as plt
import numpy as np

# Some example data to display
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)


heights = [3,1]
fig = plt.figure()
gs = fig.add_gridspec(2, 4, hspace=0, wspace=0.2,height_ratios=heights)
ax = gs.subplots(sharey=False, sharex=True)
fig.suptitle('Sharing x per column, y per row')
ax[0,0].plot(x, y)
ax[0,1].plot(x, 5*y**2, 'tab:orange')
ax[1,0].plot(x + 1, -y, 'tab:green')
ax[1,1].plot(x + 2, -y**2, 'tab:red')

# for ax in fig.get_axes():
#     ax.label_outer()

plt.show()