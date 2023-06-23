
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
import seaborn as sns
import pandas as pd
import umap


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
print('Applying UMAP')
r = 10
Umap = umap.UMAP(n_neighbors=100,n_components=r,random_state=42)

embedding = Umap.fit_transform(W_scaled.T)
print(embedding.shape)

#U_transform = Umap.transform(W_scaled.T)

#print(U_transform.shape)
#print('done')

# #############
# fig, ax = plt.subplots()
# ax.plot(e)
# ax.set_yscale('log')
# ax.set(xlabel='rank', ylabel='s',
#        title='Singular values')
# plt.show()
# ##########



#np.save('../data/data_WF_UMAP_projections_small', embedding)


# %matplotlib qt

dims = r
t = 10000

time = np.linspace(1, t, t)
neurons = np.linspace(1, dims, dims)
T, N = np.meshgrid(time, neurons)
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(T, N, embedding[:t,:dims].T, cmap = plt.cm.cividis)
fig.colorbar(surf, shrink=0.5, aspect=8)
plt.show()

###############
fig, ax = plt.subplots()
ax.plot(embedding)
ax.set(xlabel='time', ylabel='s',
       title='umap')
plt.show()
################

#time = np.linspace(1, t, t)
#neurons = np.linspace(1, dims, dims)
#T, N = np.meshgrid(time, neurons)
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')
ax.plot3D(embedding[:2000,0].T,embedding[:2000,1].T,embedding[:2000,2].T)
ax.plot3D(embedding[2000:4000,0].T,embedding[2000:4000,1].T,embedding[2000:4000,2].T)
ax.plot3D(embedding[4000:6000,0].T,embedding[4000:6000,1].T,embedding[4000:6000,2].T)
ax.plot3D(embedding[6000:8000,0].T,embedding[6000:8000,1].T,embedding[6000:8000,2].T)
ax.plot3D(embedding[8000:10000,0].T,embedding[8000:10000,1].T,embedding[8000:10000,2].T)
plt.show()


#inversion

forcast_embeddings = np.loadtxt("/home/nimbus/Documents/aditya/neuro/overschee book/examples/UMAP10_forecast_10k.csv",
                                delimiter=",")

embeddings_recon = Umap.inverse_transform(forcast_embeddings.T)

np.savetxt('../data/WF_UMAP_recon.csv',embeddings_recon,delimiter=',')



