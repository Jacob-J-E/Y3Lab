import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# Function to generate a Lorentzian peak
def lorentzian(x, x0, gamma, a):
    return a * (gamma / 2) ** 2 / ((x - x0) ** 2 + (gamma / 2) ** 2)

# Generate synthetic dataset
np.random.seed(42)
x = np.linspace(0, 100, 1000)
lorentzian_peaks = lorentzian(x, 20, 3, 10) + lorentzian(x, 60, 5, 15) + lorentzian(x, 80, 4, 8)
noise = np.random.normal(0, 1, len(x))
sine_wave = 2 * np.sin( 1* x)

data = lorentzian_peaks + noise + sine_wave

# Set parameters for peak detection
prominence_threshold = 5

# Find peaks in the observed dataset
observed_peaks, _ = find_peaks(data, prominence=prominence_threshold)

plt.plot(x,data)
plt.scatter(x[observed_peaks],data[observed_peaks])
plt.show()

# Fit Lorentzian functions to the detected peaks and calculate the goodness of fit
goodness_of_fit_threshold = 0.8
lorentzian_peak_indices = []

for peak_index in observed_peaks:
    try:
        lower = peak_index - 50
        higher = peak_index + 50
        if lower < 0:
            lower = 0
        if higher > (len(x)-1):
            higher = (len(x)-1)


        popt, pcov = curve_fit(lorentzian, x[lower: higher], data[lower: higher])
        fitted_peak = lorentzian(x[lower: higher], *popt)
        residuals = data[lower: higher] - fitted_peak
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((data[peak_index - 50: peak_index + 50] - np.mean(data[peak_index - 50: peak_index + 50]))**2)
        r_squared = 1 - (ss_res / ss_tot)

        plt.plot(x[lower: higher], data[lower: higher])
        plt.plot(x[lower: higher],)
        plt.show()
        if r_squared > goodness_of_fit_threshold:
            lorentzian_peak_indices.append(peak_index)
    except RuntimeError:
        pass

plt.plot(x,data)
plt.scatter(x[lorentzian_peak_indices],data[lorentzian_peak_indices])
plt.show()

observed_test_statistic = len(lorentzian_peak_indices)

# # Perform the permutation test
# n_permutations = 10
# permuted_test_statistics = []
# i = 0
# for _ in range(n_permutations):
#     print(i)
#     i +=1
#     # Permute the data
#     permuted_data = np.random.permutation(data)
    
#     # Find peaks in the permuted dataset
#     permuted_peaks, _ = find_peaks(permuted_data, prominence=prominence_threshold)
    
#     # Fit Lorentzian functions to the detected peaks in the permuted data and calculate the goodness of fit
#     lorentzian_permuted_peak_indices = []
#     for peak_index in permuted_peaks:
#         try:
#             lower = peak_index - 50
#             higher = peak_index + 50
#             if lower < 0:
#                 lower = 0
#             if higher > (len(x)-1):
#                 higher = (len(x)-1)
        
#             popt, pcov = curve_fit(lorentzian, x[lower: higher], permuted_data[lower: higher])
#             fitted_peak = lorentzian(x[lower: higher], *popt)
#             residuals = permuted_data[lower: higher] - fitted_peak
#             ss_res = np.sum(residuals**2)
#             ss_tot = np.sum((permuted_data[lower: higher] - np.mean(permuted_data[lower:higher]))**2)
#             r_squared = 1 - (ss_res / ss_tot)

#             if r_squared > goodness_of_fit_threshold:
#                 lorentzian_permuted_peak_indices.append(peak_index)
#         except RuntimeError:
#             pass

# # Calculate the test statistic for the permuted dataset
# permuted_test_statistic = len(lorentzian_permuted_peak_indices)
# permuted_test_statistics.append(permuted_test_statistic)

# p_value = np.sum(np.array(permuted_test_statistics) >= observed_test_statistic) / n_permutations

# print(f"P-value: {p_value}")
