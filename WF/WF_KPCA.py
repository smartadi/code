
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
from sklearn.decomposition import PCA  # to apply PCA
from sklearn.decomposition import KernelPCA

# Load spike data
WF_data = np.load('../data/data_WF_resize_10k_n100.npy')
# WF_data = np.memmap('data/data_WF_short.npy',dtype='float32',mode='r', shape=(313600, 5000))
ws = np.array(WF_data[:, :5000])
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
r = 500
print("Running KPCA full")
Kpca = KernelPCA(n_components=r, kernel='rbf',eigen_solver='auto',remove_zero_eig='True',fit_inverse_transform='True')
data_Kpca = Kpca.fit_transform(W_scaled.T)
print(data_Kpca.shape)
e = Kpca.eigenvalues_    # lam
s = Kpca.eigenvectors_   # S
s2 = Kpca.dual_coef_   # S

data_original = Kpca.inverse_transform(data_Kpca[:,:500])
print(data_original.shape)

print(W_scaled[:,2800])
print(data_original[2800,:])
re = data_original.T

ims = re[:,2800]
imr = ims.reshape(100, 100)

plt.imshow(imr)
plt.show()

print(s2.shape)
print(s.shape)

#s_r = s[:,]
#W_proj = s2

'''
print(e.shape)
#print(e)
print(s.shape)
#print(s)
E = np.diag(e)
dc = Kpca.dual_coef_
#print(dc)
#print(dc.shape)
print("KPCA done")
spike_projected = s.T@W_scaled
#############
fig, ax = plt.subplots()
ax.plot(e)
ax.set_yscale('log')
ax.set(xlabel='rank', ylabel='s',
       title='Singular values')
plt.show()

##########

#%matplotlib qt
'''
dims = r
t = 1000
time = np.linspace(1, t, t)
neurons = np.linspace(1, dims, dims)
T, N = np.meshgrid(time, neurons)

fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')

surf = ax.plot_surface(T, N, data_Kpca[:t,:dims].T, cmap = plt.cm.cividis)
# Set axes label
#ax.set_xlabel('x', labelpad=20)
#ax.set_ylabel('y', labelpad=20)
#ax.set_zlabel('z', labelpad=20)

fig.colorbar(surf, shrink=0.5, aspect=8)

plt.show()

'''
print(np.shape(spike_projected))
#np.save('../data/data_WF_KPCA_projections_small', spike_projected)
#

fig, ax = plt.subplots()
ax.plot(time,spike_projected[0:20,:t].T)
ax.set(xlabel='rank', ylabel='s',
       title='Singular values')

plt.show()

s_r = s[:,:200]
print(s.shape)
print(s_r.shape)
spike_recon = s_r@s_r.T@W_scaled
print(spike_recon.shape)
print(spike_recon[:,0])
print(W_scaled.shape)
print(W_scaled[:,0])

print(np.linalg.norm(s[0,:]))
print(np.linalg.norm(s[:,0]))
print()

ims = spike_recon[:,0]
imr = ims.reshape(100, 100)

plt.imshow(imr)
plt.show()

print(data_Kpca.shape)
print(data_Kpca[:,0])
print(spike_projected.shape)
print(spike_projected[:,0])

print(e)

'''
np.save('../data/data_WF_KPCA_projections_small2', data_Kpca.T)
np.savetxt('../data/data_WF_PCA_projections_small2.csv',data_Kpca.T,delimiter=',')
