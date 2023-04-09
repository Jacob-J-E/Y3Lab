import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

df = 22
x_ = stats.t.rvs(df=df, size=1000000)

q3_, q1_ = np.percentile(x_, [75 ,25])
iqr_ = q3_ - q1_
h = (2*iqr_)/((len(x_)**(1/3)))
bin_number_ = int((max(x_) - min(x_))/h)

freq_, bin_edges_ = np.histogram(x_,density=True, bins = bin_number_)
mid_bin_edges_ = (bin_edges_[1:] + bin_edges_[:-1])/2

alpha_5 = 0.05
alpha_1 = 0.01
alpha_values = np.linspace(0,1,10000)

critical_t_5_upper = stats.t.ppf(q=(1-(alpha_5/2)), df= df)
critical_t_1_upper = stats.t.ppf(q=(1 - (alpha_1/2)), df= df)
critical_t_5_lower = stats.t.ppf(q=alpha_5/2, df= df)
critical_t_1_lower = stats.t.ppf(q=alpha_1/2, df= df)

print(critical_t_5_upper)
print(critical_t_5_lower)

plt.hist(x_, density=True, edgecolor='black', bins=bin_number_)
plt.axvline(critical_t_5_upper, color = 'purple',label = f'Critical T-Value at 95% confidence (Two-Tailed) T: [{critical_t_5_upper}]')
plt.axvline(critical_t_5_lower,color = 'purple',)
plt.axvline(critical_t_1_upper, color = 'blue', label = f'Critical T-Value at 99% confidence (Two-Tailed) T: [{critical_t_1_upper}]')
plt.axvline(critical_t_1_lower, color = 'blue')
plt.fill_between(mid_bin_edges_,freq_,0,where=(mid_bin_edges_>=critical_t_5_lower) & (mid_bin_edges_<=critical_t_5_upper),color='red', label = r'Reject $H_{0}$ at 95% confidence', alpha=0.5)
plt.show()