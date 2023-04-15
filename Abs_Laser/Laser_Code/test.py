import matplotlib.pyplot as plt
import numpy as np

# Some example data to display
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)


width = [2,1]
fig = plt.figure()
gs = fig.add_gridspec(4, 2, hspace=0.5, wspace=0,width_ratios=width)
ax = gs.subplots(sharey=False, sharex=False)
fig.suptitle('Sharing x per column, y per row')
ax[0,0].plot(x, y)
ax[0,0].set_title('woah',loc='left')
ax[0,1].plot(x, y)
ax[1,0].set_title('woah',loc='left')
ax[1,0].plot(x, 5*y**2, 'tab:orange')
ax[1,1].plot(x, 5*y**2, 'tab:orange')
ax[2,0].plot(x + 1, -y, 'tab:green')
ax[2,1].plot(x + 1, -y, 'tab:green')
ax[3,1].plot(x + 2, -y**2, 'tab:red')
ax[3,0].plot(x + 2, -y**2, 'tab:red')

# for ax in fig.get_axes():
#     ax.label_outer()

plt.show()