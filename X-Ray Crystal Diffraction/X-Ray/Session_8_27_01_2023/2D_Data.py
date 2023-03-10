import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import argrelextrema
from skimage.feature import peak_local_max

def fourier_clean_2D(data,threshold):
    """
    FT algorithm to clean data.
    Takes FFT of y-data, sets high freq data to 0, takes inverse FFT
    Input y-data, x-data, threshold.
    """
    
    # size = len(x_data)
    # fft = np.fft.fft2(data,size)
    fft = np.fft.fft2(data)
    # abs_fft = fft * np.conj(fft)/size
    abs_fft = fft * np.conj(fft)
    signal_threshold = threshold
    abs_fft_zero_arr = abs_fft > signal_threshold #array of 0 and 1 to indicate where data is greater than threshold
    fhat_clean = abs_fft_zero_arr * fft #used to retrieve the signal
    signal_filtered = np.fft.ifft2(fhat_clean) #inverse fourier transform

    return np.real(signal_filtered)

# Load in data.
data = pd.read_csv(r"X-Ray\Data\28-01-2023\2D Data NaCl Powder Cu Source.csv",skiprows=0,header=None)

# Remove NAN values and transpose data so angle is on the x-axis.
data.replace("#NAME?",0,inplace=True)
data = np.array(data,dtype=float)
data = np.rot90(data)

# print(data)
# print(data[0])
# print(data[1])
# print(data[2])
# print(data[2])
# print("AAA",data[2][4])



# print(np.shape(data))
# print(np.shape(data[0]))
# print(np.shape(data[1]))


# Maxima finder

# xy = peak_local_max(data, min_distance=1,threshold_abs=3)
# print("splice",xy[1][0])

# x_val = []
# y_val = []
# for i in range(0,len(xy)):
#     x_val.append(xy[i][0])
#     y_val.append(xy[i][1])
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import matplotlib.pyplot as pp


# def detect_peaks(image):
#     """
#     Takes an image and detect the peaks usingthe local maximum filter.
#     Returns a boolean mask of the peaks (i.e. 1 when
#     the pixel's value is the neighborhood maximum, 0 otherwise)
#     """

#     # define an 8-connected neighborhood
#     neighborhood = generate_binary_structure(2,2)

#     #apply the local maximum filter; all pixel of maximal value 
#     #in their neighborhood are set to 1
#     local_max = maximum_filter(image, footprint=neighborhood)==image
#     #local_max is a mask that contains the peaks we are 
#     #looking for, but also the background.
#     #In order to isolate the peaks we must remove the background from the mask.

#     #we create the mask of the background
#     background = (image==0)

#     #a little technicality: we must erode the background in order to 
#     #successfully subtract it form local_max, otherwise a line will 
#     #appear along the background border (artifact of the local maximum filter)
#     eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

#     #we obtain the final mask, containing only peaks, 
#     #by removing the background from the local_max mask (xor operation)
#     detected_peaks = local_max ^ eroded_background

#     return detected_peaks
# peaks = detect_peaks(data)
# print(peaks)


# Generate data for theoretical 2D plot.
angle = np.arange(0,90,0.01)
energy = np.arange(0,100,0.1)
sin_angle = np.sin(angle * np.pi / 180)
A_mesh, E_mesh = np.meshgrid(sin_angle,energy)
Z = E_mesh * sin_angle

# Fourier Transform
data_clean = fourier_clean_2D(data,threshold=0.8e4)
# dx = np.diff(data)[0]
# freqs = np.fft.fftfreq(len(data), d=dx)

# Generate colormap numerical ranges.
combined_data = np.array([data,data_clean])
_min, _max = np.amin(combined_data), np.amax(combined_data)


# Plot data.
fig,ax = plt.subplots(1,3)
# data_2D = ax[0].imshow(data,extent=(0,90,0,150),aspect=0.5, vmin = _min, vmax = _max)
data_2D = ax[0].imshow(data,extent=(0,90,0,150),aspect=0.5, vmin = _min, vmax = _max)
# ax[0].scatter(x_val,y_val,color='red')
data_fft = ax[1].imshow(data_clean,extent=(0,90,0,150),aspect=0.5, vmin = _min, vmax = _max)
data_differnce = ax[2].imshow(data-data_clean,extent=(0,90,0,150),aspect=0.5, vmin = _min, vmax = _max)
ax[0].set_title("2D Energy-Angle Data")
ax[1].set_title("Fourier Cleaned Energy-Angle Data")
ax[2].set_title("Cleaned Data 'Differnce'")
# ax[0].contour(Z,extent=(min(angle),max(angle),min(energy),max(energy)),label="Bragg Contour")
# ax[3].contourf(Z,extent=(min(angle),max(angle),min(energy),max(energy)),label=r"m^{th} Bragg Order Ranges")
# ax[3].set_aspect('equal')

for ax in ax:
    ax.set_xlabel("Angle (Degrees)")
    ax.set_ylabel("Energy Channel (Arb.)")


fig.colorbar(data_2D)
plt.show()
