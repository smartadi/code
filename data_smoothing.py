import numpy as np
import matplotlib.pyplot as plt
# Make numpy print 4 significant digits for prettiness
np.set_printoptions(precision=4, suppress=True)
np.random.seed(5) # To get predictable random numbers

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

import matplotlib.animation as animation
from sklearn import datasets  # to retrieve the iris Dataset
import pandas as pd  # to load the dataframe
from sklearn.preprocessing import StandardScaler  # to standardize the features
from sklearn.decomposition import PCA  # to apply PCA
import seaborn as sns  # to plot the heat maps#float32


##########
# Load spike data
spike_data = np.load('data/spikes_v1_clean.npy')
print(np.shape(spike_data))
time = 100000
spike_data_short = spike_data[:,0:time]
# short time span
t = np.shape(spike_data_short)[1]
n = np.shape(spike_data_short)[0]

n_y = np.arange(n)
t_x = np.arange(t)




y_spike = spike_data_short
#t_full = np.linspace(1, t, t)
#t_x = np.arange(length)
#n_y = np.shape(y_spike)[0]
print(np.shape(y_spike))

#plt.bar(t_x,y_spike[0,0:])
############
def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))
def sigma2fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))
##############
# tunning parameter
FWHM = 20
sigma = fwhm2sigma(FWHM)
#x_position = 100 # 14th point
#kernel_at_pos_n = np.exp(-(t_x - x_position) ** 2 / (2 * sigma ** 2))
#kernel_at_pos_n = kernel_at_pos_n / sum(kernel_at_pos_n)
############
smoothed_vals_spike_full = np.zeros(y_spike.shape)

for nn in n_y:
    # smoothed_vals_spike = np.zeros(y_spike[nn,:].shape)
    print(nn)

    for x_position in t_x:
        kernel = np.exp(-(t_x - x_position) ** 2 / (2 * sigma ** 2))
        kernel = kernel / sum(kernel)
        smoothed_vals_spike_full[nn, x_position] = sum(y_spike[nn, :] * kernel)
#################
np.save('data/data_smooth_W20L100k', smoothed_vals_spike_full)


