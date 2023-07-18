# sample scripts to resize image


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
import cv2

# Load spike data
WF_data = np.memmap('../data/data_WF_short.npy', mode='r', shape=(313600, 5000))
W_short = np.array(WF_data)
ims = W_short.reshape(560,560,5000)
ims_re = np.empty((240,240,5000))
print(ims.shape)

#
for i in range(W_short.shape[1]):
    ims_re[:,:,i] = cv2.resize(ims[:,:,i],(240,240))

data = ims_re.reshape(-1, 5000)
print(ims_re.shape)
np.save('data/data_WF_resize', data)


