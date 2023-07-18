import matplotlib.pyplot as plt
import numpy as np

import skfda
from skfda.misc.hat_matrix import (
    KNeighborsHatMatrix,
    LocalLinearRegressionHatMatrix,
    NadarayaWatsonHatMatrix,
)
from skfda.misc.kernels import uniform
from skfda.preprocessing.smoothing import KernelSmoother
from skfda.preprocessing.smoothing.validation import SmoothingParameterSearch
from sklearn.preprocessing import StandardScaler  # to standardize the features
import pandas as pd

# Load spike data
spike_data = np.load('../data/spikes_v1_clean.npy')
spike_data_t = spike_data[:,0:999]
print(np.shape(spike_data))

# short time span
t = np.shape(spike_data)[1]
n = np.shape(spike_data)[0]

scalar = StandardScaler()
scaled_data = pd.DataFrame(scalar.fit_transform(spike_data)) #scaling the data
print(scaled_data.shape)



# scaled data or non scaled data
length = 1000
nn = 1
y_spike_full = pd.DataFrame.to_numpy(scaled_data)

y_spike = spike_data[nn-1,0:length]
t_full = np.linspace(1, t, t)
t_x = np.arange(length)
n_y = np.shape(y_spike)[0]

plt.bar(t_x,y_spike)

t = np.shape(spike_data)[1]
n = np.shape(spike_data)[0]
print(t)
print(n)


def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))

def sigma2fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))


FWHM = 20
sigma = fwhm2sigma(FWHM)
t=10000
grid_points = np.linspace(1,t,t)
fd = skfda.FDataGrid(
    data_matrix=spike_data[:,:t],
    grid_points=grid_points,
)
spike_full = KernelSmoother(
    kernel_estimator=NadarayaWatsonHatMatrix(bandwidth=20),
).fit_transform(fd)


t=1000
plt.plot(spike_data[200,:t])
plt.plot(spike_full.data_matrix[200][:t])