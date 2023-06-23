
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
from sklearn.decomposition import PCA  # to apply PCA

# Load spike data
WF_data = np.load('../data/data_WF_resize_5k_norm.npy')
# WF_data = np.memmap('data/data_WF_short.npy',dtype='float32',mode='r', shape=(313600, 5000))
ws = np.array(WF_data[:, :1000])
w = np.array(WF_data)

W_short = ws.astype(np.float32)
W = w.astype(np.float32)


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
[u, s, v] = pca_scale(W_scaled)
r = 500
u_r = u[:, :r]
s_r = s[:r]
D = np.diag(s_r)
########################

W_proj = u_r.T@W_scaled
# W_proj = u_r.T@W_short
# W_proj = np.linalg.inv(S)@u_r.T@W_short
# np.save('data/data_WF_scaled5', W_scaled)
# np.save('data/data_WF_projected_scaled5', W_proj)
# print(W_proj[:,0])

#############
fig, ax = plt.subplots()
ax.plot(s)
ax.set_yscale('log')
ax.set(xlabel='rank', ylabel='S',
       title='Singular values')
plt.show()

##########
dims = 50
t = 5000
time = np.linspace(1, t, t)
d = np.linspace(1, dims, dims)
T, N = np.meshgrid(time, d)

fig = plt.figure(figsize=(12, 10))
ax = plt.axes(projection='3d')

surf = ax.plot_surface(T, N, W_proj[0:dims, :], cmap=plt.cm.cividis)
# Set axes label
# ax.set_xlabel('x', labelpad=20)
# ax.set_ylabel('y', labelpad=20)
# ax.set_zlabel('z', labelpad=20)

fig.colorbar(surf, shrink=0.5, aspect=8)
plt.show()


@jit(target_backend='cuda')
def pca_recon(a, ur):
    up = ur @ ur.T
    b = up @ a

    return b


'''
W_rcon = PCA_recon(W_short,u_r)

print(W_rcon[:,0])
ims = W_rcon[:,0]
imr = ims.reshape(560, 560)

plt.imshow(imr)
plt.show()

print()
'''

# print(u_r[:,0])

W_proj_scaled, _, _ = scaling_numba1(W_proj)

dims = 50
t = 5000
time = np.linspace(1, t, t)
d = np.linspace(1, dims, dims)
T, N = np.meshgrid(time, d)

fig = plt.figure(figsize=(12, 10))
ax = plt.axes(projection='3d')

surf = ax.plot_surface(T, N, W_proj_scaled[0:dims, :], cmap=plt.cm.cividis)
fig.colorbar(surf, shrink=0.5, aspect=8)
plt.show()

fig, ax = plt.subplots()
ax.plot(time,W_proj[0,:].T)
# ax.set_yscale('log')
ax.set(xlabel='time', ylabel='mode',
       title='mode')
plt.show()



#######################
r = 500
pca = PCA(n_components = r)

# X = USV'
data_pca = pca.fit_transform(W_scaled) # = US

c = pca.components_      # V
si = pca.singular_values_ # S
Si = np.diag(si)

fig, ax = plt.subplots()
ax.plot(si)

ax.set(xlabel='rank', ylabel='S',
       title='Singular values')
plt.show()

#########
# X_proj = U'X
# X_proj = inv(S)S'U'X
spike_projected = np.linalg.inv(Si)@data_pca.T@W_scaled
##########

t=5000
dims=10
time = np.linspace(1, t, t)
d = np.linspace(1, dims, dims)
T, N = np.meshgrid(time, d)

fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')

surf = ax.plot_surface(T, N, spike_projected[0:dims,:], cmap = plt.cm.cividis)

fig.colorbar(surf, shrink=0.5, aspect=8)

plt.show()
