import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from numba import jit, prange

import reservoirpy as rpy
from sklearn.preprocessing import normalize
rpy.set_seed(42)

from reservoirpy.observables import nrmse, rsquare

############
@jit(target_backend='cuda')
def scaling_numba1(a):
    mean = np.zeros((1, a.shape[1]))
    for j in prange(a.shape[1]):
        for i in prange(a.shape[0]):
            mean[0, j] = mean[0, j] + a[i, j]

        mean[0, j] = mean[0, j] / a.shape[0]

    std = np.zeros((1, a.shape[1]))
    for j in prange(a.shape[1]):
        ss = float(0)
        for i in prange(a.shape[0]):
            ss = ss + (a[i, j] - mean[0, j]) ** 2
        std[0, j] = np.sqrt(ss / a.shape[0])

    for j in prange(a.shape[1]):
        for i in prange(a.shape[0]):
            a[i, j] = (a[i, j] - mean[0, j]) / std[0, j]

    return a, mean, std


WF_data = np.load('../data/data_WF_PCA_projections_small.npy')

print(WF_data.shape)

print(WF_data[:,0])

# WF_scaled,_,_ = scaling_numba1(WF_data)
#
# print(WF_scaled[:,0])
#
# np.linalg.norm(WF_scaled[:,0])

WF_data_scaled = WF_data / np.linalg.norm(WF_data,2,0)


fig = plt.figure(figsize=(12,10))
plt.plot(np.arange(0, 1000), WF_data_scaled[:, :1000].T, label="Training data")
plt.show()


WF_trainX = WF_data_scaled[:10,:1000]
WF_trainY = WF_data_scaled[:10,1:1001]
print(WF_trainX.shape)
print(WF_trainY.shape)


WF_testX = WF_data_scaled[:10,1000:2000]
WF_testY = WF_data_scaled[:10,1001:2001]



print(WF_testX.shape)
print(WF_testY.shape)


