## Here we repeat the PCA on WF data for low resolution images 100*100 on 10k images from original dataset.
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
from sklearn.decomposition import PCA  # to apply PCA

# Load spike data image vector(100*100) by time(10000)
WF_data = np.load('../data/data_WF_resize_10k_n100.npy')

t = 1000
ws = np.array(WF_data[:, :t])
w = np.array(WF_data)

W_short = ws.astype(np.float32)
W = w.astype(np.float32)
print("data loaded")
print(w.shape)
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


W_scaled, _, _ = scaling_numba1(W_short)
[u, s, v] = pca_scale(W_scaled)

# rank of reconstruction
r = 500
u_r = u[:, :r]
s_r = s[:r]
v_r = v[:r,:]            # The temporal components we need
D = np.diag(s_r)
US = u_r*s_r
########################

W_proj = u_r.T@W_scaled
print(W_proj.shape)

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

surf = ax.plot_surface(T, N, W_proj[0:dims, :t], cmap=plt.cm.cividis)
# Set axes label
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)
ax.set_title('projected dimensions')
fig.colorbar(surf, shrink=0.5, aspect=8)
plt.show()
############################
dims = 5
t = 1000
fig, ax = plt.subplots()
ax.plot(time,v_r[0:dims, :t].T)
ax.set(xlabel='time', ylabel='temoral modes',
       title='TV_r')
plt.show()

@jit(target_backend='cuda')
def pca_recon(a, ur):
    up = ur @ ur.T
    b = up @ a

    return b


# uncomment for reconstruction
# W_rcon = PCA_recon(W_short,u_r)
#
# print(W_rcon[:,0])
# ims = W_rcon[:,0]
# imr = ims.reshape(560, 560)
#
# plt.imshow(imr)
# plt.show()
#
# print()


# print(u_r[:,0])
'''
W_proj_scaled, _, _ = scaling_numba1(W_proj)

dims = 50
t = 5000
time = np.linspace(1, t, t)
d = np.linspace(1, dims, dims)
T, N = np.meshgrid(time, d)

fig = plt.figure(figsize=(12, 10))
ax = plt.axes(projection='3d')

surf = ax.plot_surface(T, N, W_proj_scaled[0:dims, :t], cmap=plt.cm.cividis)
fig.colorbar(surf, shrink=0.5, aspect=8)
plt.show()

fig, ax = plt.subplots()
ax.plot(time,W_proj[0,:t].T)
# ax.set_yscale('log')
ax.set(xlabel='time', ylabel='mode',
       title='mode')
plt.show()
'''


#######################
'''
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

surf = ax.plot_surface(T, N, spike_projected[0:dims,:t], cmap = plt.cm.cividis)

fig.colorbar(surf, shrink=0.5, aspect=8)

plt.show()
'''



'''
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
dims = 500
t=1000
time = np.linspace(1, t, t)
neurons = np.linspace(1, dims, dims)


T, N = np.meshgrid(time, neurons)
#surf = ax.plot_surface(T, N, spike_data[:,0:1000], cmap = plt.cm.cividis)
surf = ax.plot_surface(T, N, W_proj_scaled[:dims,:t], cmap = plt.cm.cividis)

fig.colorbar(surf, shrink=0.5, aspect=8)

# ==============
# Second subplot
# ==============
# set up the axes for the second plot
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_title('Reconstructed data')

# plot a 3D wireframe like in the example mplot3d/wire3d_demo
surf = ax.plot_surface(T, N, W_proj[:dims,:t], cmap = plt.cm.cividis)
fig.colorbar(surf, shrink=0.5, aspect=8)

plt.show()


print(np.shape(W_proj))

'''
# np.save('../data/data_WF_PCA_temporal_100res_10k', vr)
# np.save('../data/data_WF_US_small', US)
# np.savetxt('../data/data_WF_US_small.csv', US,delimiter=',')
