
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn import datasets  # to retrieve the iris Dataset
import pandas as pd  # to load the dataframe
from sklearn.preprocessing import StandardScaler  # to standardize the features
from sklearn.decomposition import PCA  # to apply PCA
import seaborn as sns  # to plot the heat maps#float32



spike_data = np.load('data/spikes_v1_clean.npy')                        #



print(np.shape(spike_data))

time = np.shape(spike_data)[1]
n = np.shape(spike_data)[0]



scalar = StandardScaler()
scaled_data = pd.DataFrame(scalar.fit_transform(scalar)) #scaling the data
print(scaled_data)