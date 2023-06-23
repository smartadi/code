import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn import datasets  # to retrieve the iris Dataset
import pandas as pd  # to load the dataframe
from sklearn.preprocessing import StandardScaler  # to standardize the features
from sklearn.decomposition import PCA  # to apply PCA
import seaborn as sns  # to plot the heat maps#float32
from numba import jit, cuda
from numba import jit, prange




# Load spike data
WF_data = np.memmap('data/data_WF_short.npy',dtype='float32',mode='r', shape=(313600, 5000))
W_short = np.array(WF_data[:,:1000])
W = np.array(WF_data)


@jit(target_backend='cuda')
def scaling_numba1(a):
    mean = np.zeros((1, a.shape[1]))

    for j in prange(a.shape[1]):
        for i in prange(a.shape[0]):
            mean[0, j] = mean[0, j] + a[i, j]

        mean[0, j] = mean[0, j] / a.shape[0]

    std = np.zeros((1, a.shape[1]))
    for j in prange(a.shape[1]):
        for i in prange(a.shape[0]):
            std[0, j] = np.sqrt((a[i, j] - mean[0, j]) ** 2 / a.shape[0])

    for j in prange(a.shape[1]):
        a[:, j] = (a[:, j] - mean[0, j]) / std[0, j]

    return a


W_scaled = scaling_numba1(W_short)


@jit(target_backend='cuda')
def PCA_scale(a):
    u, s, v = np.linalg.svd(a, full_matrices=False)
    return u, s, v


[u,s,v] = PCA_scale(W_scaled)

r=100
u_r = u[:,:r]
s_r = s[:r]
S = np.diag(s_r)

#_proj = u_r.T@W_short
W_proj = u_r.T@W_scaled

#np.save('data/data_WF_scaled5', W_scaled)
#np.save('data/data_WF_projected_scaled5', W_proj)


