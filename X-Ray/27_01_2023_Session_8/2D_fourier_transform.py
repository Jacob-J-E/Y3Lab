import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from scipy import fftpack

# Load in data.
data = pd.read_csv(r"X-Ray\Data\28-01-2023\2D Data NaCl Powder Cu Source.csv",skiprows=0,header=None)

# Remove NAN values and transpose data so angle is on the x-axis.
data.replace("#NAME?",0,inplace=True)
# print(data)

plt.figure()
plt.imshow(np.array(data.to_numpy()))
plt.title('Image')
plt.show()

im_fft = fftpack.fft2(data)

# Show the results

def plot_spectrum(im_fft):
    from matplotlib.colors import LogNorm
    # A logarithmic colormap
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()

plt.figure()
plot_spectrum(im_fft)
plt.title('Fourier transform')
#plt.show()

# In the lines following, we'll make a copy of the original spectrum and
# truncate coefficients.

# Define the fraction of coefficients (in each direction) we keep
keep_fraction = 0.2

# Call ff a copy of the original transform. Numpy arrays have a copy
# method for this purpose.
im_fft2 = im_fft.copy()

# Set r and c to be the number of rows and columns of the array.
r, c = im_fft2.shape

# Set to zero all rows with indices between r*keep_fraction and
# r*(1-keep_fraction):
im_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0

# Similarly with the columns:
im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0

plt.figure()
plot_spectrum(im_fft2)
plt.title('Filtered Spectrum')
#plt.show()

im_new = fftpack.ifft2(im_fft2).real

print(im_new)

plt.figure()
plt.imshow(im_new)
plt.title('Reconstructed Image')
plt.show()