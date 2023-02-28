import numpy as np
import scipy.optimize as spo

class Inital_y_guess:
    def __init__(self,combined_x,combined_y) -> None:
        self.combined_x = combined_x
        self.combined_y = combined_y


    def split_data_into_half(self):
        combined_x = self.combined_x
        combined_y = self.combined_y
        med = np.median(combined_x)
        mid_shift = med + 3
        medium_range_y = combined_y[(combined_x > med - 0.5) & (combined_x < med + 0.5)]
        medium_range_y_shift = combined_y[(combined_x > mid_shift - 0.5) & (combined_x < mid_shift + 0.5)]
        combine = list(medium_range_y_shift) + list(medium_range_y)
        combine_sorted = sorted(combine)

        fh_med = combine_sorted[:int(len(combine_sorted)/2)]
        sh_med = combine_sorted[int(len(combine_sorted)/2):]

        return fh_med,sh_med
    
    def fitting_gaussian(self, data):
        def gaussian(x, a, b, c):
            return (a * np.exp(-((x - b) ** 2) / (2 * c ** 2)))

        # first histogram - lower

        num_of_bins = 20

        hist, bin_edges = np.histogram(data, bins=num_of_bins)

        center_of_bins = (bin_edges[:-1] + bin_edges[1:]) / 2



        first_index = np.where(hist == max(hist))
        first_index = first_index[0][0]

        first_mean_guess = center_of_bins[first_index]
        width = (center_of_bins[1]- center_of_bins[0])

        guess = [max(hist),first_mean_guess,width/4]
        params, cov = spo.curve_fit(gaussian,center_of_bins,hist,guess)

        return params, cov 


    def guess_y(self):
        fh_med,sh_med = self.split_data_into_half()

        params_fh, cov_fh = self.fitting_gaussian(fh_med)
        params_sh, cov_sh = self.fitting_gaussian(sh_med)

        return params_fh, cov_fh, params_sh, cov_sh

