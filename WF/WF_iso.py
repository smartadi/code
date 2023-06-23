
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
from sklearn.decomposition import PCA  # to apply PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import FactorAnalysis
from sklearn.manifold import Isomap



# Load spike data
WF_data = np.load('../data/data_WF_resize_10k_n100.npy')
# WF_data = np.memmap('data/data_WF_short.npy',dtype='float32',mode='r', shape=(313600, 5000))
ws = np.array(WF_data[:, :1000])
w = np.array(WF_data)

W_short = ws.astype(np.float32)
W = w.astype(np.float32)

print("data loaded")

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


################


@jit(target_backend='cuda')
def pca_scale(a):
    uu, ss, vvh = np.linalg.svd(a, full_matrices=False)
    return uu, ss, vvh


#######################
W_scaled, _, _ = scaling_numba1(W)
######################
print('Applying Isomap')

r = 500
Iso = Isomap(n_components = r)
iso = Iso.fit_transform(W_scaled) # = US
emb = Iso.embedding_
Kpca = Iso.kernel_pca_
s = Kpca.eigenvectors_ # S
print(s.shape)
e = Kpca.eigenvalues_ # S
print(e.shape)
fig, ax = plt.subplots()
ax.plot(e)
ax.set(xlabel='rank', ylabel='S',
       title='eigen values')
plt.show()

print('isomapped')
spike_projected = s.T@W_scaled
#############
fig, ax = plt.subplots()
ax.plot(e)
ax.set_yscale('log')
ax.set(xlabel='rank', ylabel='s',
       title='Singular values')
plt.show()
##########

# %matplotlib qt
dims = r
t = 1000
time = np.linspace(1, t, t)
neurons = np.linspace(1, dims, dims)
T, N = np.meshgrid(time, neurons)
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(T, N, spike_projected[0:dims,:t], cmap = plt.cm.cividis)
fig.colorbar(surf, shrink=0.5, aspect=8)
plt.show()

print(np.shape(spike_projected))
np.save('../data/data_WF_ISO_projections_small', spike_projected)