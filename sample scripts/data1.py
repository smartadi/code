from numba import jit, cuda
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import matplotlib.animation as animation
from sklearn import datasets  # to retrieve the iris Dataset
import pandas as pd  # to load the dataframe
from sklearn.preprocessing import StandardScaler  # to standardize the features
from sklearn.decomposition import PCA  # to apply PCA
import seaborn as sns  # to plot the heat maps#float32


def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))

def sigma2fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))

@jit(target_backend='cuda')
def smooth(a):
    smoothed = np.zeros(a.shape)
    f = 20
    sigma = f / np.sqrt(8 * np.log(2))

    t = np.shape(a)[1]
    n = np.shape(a)[0]

    t_x = np.arange(t)
    n_y = np.arange(n)

    for nn in n_y:
        # smoothed_vals_spike = np.zeros(y_spike[nn,:].shape)
        print(nn)

        for x_position in t_x:
            kernel = np.exp(-(t_x - x_position) ** 2 / (2 * sigma ** 2))
            kernel = kernel / sum(kernel)
            smoothed[nn, x_position] = sum(a[nn, :] * kernel)
    return smoothed
##############


##########
# Load spike data
if __name__=="__main__":
    spike_data = np.load('../data/spikes_v1_clean.npy')
    print(np.shape(spike_data))

    time = 100000

    spike_data_short = spike_data[:, 0:time]

    y_spike = spike_data_short

    start = timer()
    smoothed_vals_spike_full = smooth(y_spike)
    print("without GPU:", timer()-start)



    np.save('../data/data_smooth_W20L100k', smoothed_vals_spike_full)


    '''
    t = np.shape(y_spike)[1]
    n = np.shape(y_spike)[0]
    fig = plt.figure(figsize = (12,10))
    ax = plt.axes(projection='3d')

    time = np.linspace(1, t, t)
    neurons = np.linspace(1, n, n)
    T, N = np.meshgrid(time, neurons)

    surf = ax.plot_surface(T, N, smoothed_vals_spike_full, cmap = plt.cm.cividis)

    fig.colorbar(surf, shrink=0.5, aspect=8)

    plt.show()'''