import numpy as np
import matplotlib.pyplot as plt
import math

"""
2dsin(theta) = mlambda
2dsin(theta) = mhc/E
Esin(theta) = m(hc/2d)
"""

angle = np.arange(0,90,0.01)
energy = np.arange(0,100,0.1)
sin_angle = np.sin(angle * np.pi / 180)

A_mesh, E_mesh = np.meshgrid(sin_angle,energy)
Z = E_mesh * sin_angle

fig, ax = plt.subplots(1,2)

ax[0].contour(Z,extent=(min(angle),max(angle),min(energy),max(energy)),label="Bragg Contour")
ax[1].contourf(Z,extent=(min(angle),max(angle),min(energy),max(energy)),label=r"m^{th} Bragg Order Ranges")

ax[0].set_title("Bragg Contours")
ax[1].set_title(r"$m^{th}$ Order Bragg Ranges")

for ax in ax:
    ax.set_xlabel("Angle (Degrees)")
    ax.set_ylabel(r"Energy (Arb.)")
plt.tight_layout()
plt.savefig(r"X-Ray\Plots and Figs\Bragg_contour")
plt.show()