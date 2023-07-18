# Converts npy to csv for use in matlab

import numpy as np


data = np.load('../data/data_WF_resize_10k_n100.npy')
np.savetxt('../data/data_WF_resize_10k_n100.csv',data,delimiter=',')

'''
# PCA
data = np.load('../data/data_WF_PCA_projections_small.npy')
np.savetxt('../data/data_WF_PCA_projections_small.csv',data,delimiter=',')




# KPCA
data = np.load('../data/data_WF_KPCA_projections_small.npy')
np.savetxt('../data/data_WF_KPCA_projections_small.csv',data,delimiter=',')


# UMAP
data = np.load('../data/data_WF_UMAP_projections_small.npy')
np.savetxt('../data/data_WF_UMAP_projections_small.csv',data,delimiter=',')


data = np.load('../data/smooth_spike_PCA_proj_10k.npy')
np.savetxt('../data/smooth_spike_PCA_proj_10k.csv',data,delimiter=',')

data = np.load('../data/smooth_spike_KPCA_proj_10k.npy')
np.savetxt('../data/smooth_spike_KPCA_proj_10k.csv',data,delimiter=',')

data = np.load('../data/smooth_spike_isomap_proj_10k.npy')
np.savetxt('../data/smooth_spike_isomap_proj_10k.csv',data,delimiter=',')

data = np.load('../data/smooth_spike_UMAP_proj_10k.npy')
np.savetxt('../data/smooth_spike_UMAP_proj_10k.csv',data,delimiter=',')
'''