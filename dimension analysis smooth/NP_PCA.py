import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
from sklearn.decomposition import PCA  # to apply PCA

# Load spike data
NP_data = np.load('../data/data_smooth_W20L50k.npy')

NPS = np.array(NP_data[:, :1000])
NP = np.array(NP_data)
print(NP_data.shape)
print(NP_data.dtype)

#W_short = ws.astype(np.float32)
#W = w.astype(np.float32)
#print("data loaded")

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
NP_scaled, _, _ = scaling_numba1(NP)
[u, s, v] = pca_scale(NP_scaled)
r = 200
u_r = u[:, :r]
s_r = s[:r]
D = np.diag(s_r)
########################

NP_proj = u_r.T@NP_scaled

print(NP_proj.shape)

#############
fig, ax = plt.subplots()
ax.plot(s)
ax.set_yscale('log')
ax.set(xlabel='rank', ylabel='S',
       title='Singular values')
plt.show()

##########
dims = 50
t = 1000
time = np.linspace(1, t, t)
d = np.linspace(1, dims, dims)
T, N = np.meshgrid(time, d)

fig = plt.figure(figsize=(12, 10))
ax = plt.axes(projection='3d')

surf = ax.plot_surface(T, N, NP_proj[0:dims, :t], cmap=plt.cm.cividis)
# Set axes label
# ax.set_xlabel('x', labelpad=20)
# ax.set_ylabel('y', labelpad=20)
# ax.set_zlabel('z', labelpad=20)
ax.set_title('projected dimensions')
fig.colorbar(surf, shrink=0.5, aspect=8)
plt.show()


@jit(target_backend='cuda')
def pca_recon(a, ur):
    up = ur @ ur.T
    b = up @ a

    return b
NP_rcon = pca_recon(NP,u_r)

#NP_proj_scaled, _, _ = scaling_numba1(NP_proj)

from mpl_toolkits.mplot3d.axes3d import get_test_data
from matplotlib import cm
# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=plt.figaspect(0.5))

# =============
# First subplot
# =============
# set up the axes for the first plot
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_title('Original data')
time = np.linspace(1, 5000, 5000)
neurons = np.linspace(1, 336, 336)


T, N = np.meshgrid(time, neurons)
#surf = ax.plot_surface(T, N, spike_data[:,0:1000], cmap = plt.cm.cividis)
surf = ax.plot_surface(T, N, NP[:,0:5000], cmap = plt.cm.cividis)

fig.colorbar(surf, shrink=0.5, aspect=8)

# ==============
# Second subplot
# ==============
# set up the axes for the second plot
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_title('Reconstructed data')

# plot a 3D wireframe like in the example mplot3d/wire3d_demo
surf = ax.plot_surface(T, N, NP_rcon[:,0:5000], cmap = plt.cm.cividis)
fig.colorbar(surf, shrink=0.5, aspect=8)

plt.show()



print(np.shape(NP_proj))
np.save('../data/data_NP_PCA_projections', NP_proj)
np.save('../data/NP_PCA_map', u_r)

np.savetxt('../data/NP_PCA_map.csv',u_r,delimiter=',')
