
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
from sklearn.decomposition import PCA  # to apply PCA
from sklearn.decomposition import KernelPCA

# Load spike data
spike_projected = np.load('../data/data_WF_KPCA_projections_small.npy')


##########

#%matplotlib qt
dims = 100
t = 1000
time = np.linspace(1, t, t)
neurons = np.linspace(1, dims, dims)
T, N = np.meshgrid(time, neurons)

fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')

surf = ax.plot_surface(T, N, spike_projected[0:dims,:t], cmap = plt.cm.cividis)
# Set axes label
#ax.set_xlabel('x', labelpad=20)
#ax.set_ylabel('y', labelpad=20)
#ax.set_zlabel('z', labelpad=20)

fig.colorbar(surf, shrink=0.5, aspect=8)

plt.show()


#

fig, ax = plt.subplots()
ax.plot(time,spike_projected[0:50,:t].T)
ax.set(xlabel='rank', ylabel='s',
       title='Singular values')

plt.show()
################################


spike_projected_u = np.load('../data/data_WF_UMAP_projections_small.npy')
print(spike_projected_u.shape)


dims = 50
t = 1000

'''
time = np.linspace(1, t, t)
neurons = np.linspace(1, dims, dims)
T, N = np.meshgrid(time, neurons)

fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')

surf = ax.plot_surface(T, N, spike_projected_u[:t, :dims], cmap = plt.cm.cividis)


fig.colorbar(surf, shrink=0.5, aspect=8)

plt.show()
'''

#

fig, ax = plt.subplots()
ax.plot(time,spike_projected_u[:t,0:2])
ax.set(xlabel='rank', ylabel='s',
       title='Singular values')

plt.show()
################################

fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')
ax.plot3D(spike_projected_u[:,0].T,spike_projected_u[:,1].T,spike_projected_u[:,2].T)
plt.show()


################################
fig, ax = plt.subplots()
ax.plot(spike_projected_u[0:1000,0],spike_projected_u[0:1000,2],'r*')
ax.plot(spike_projected_u[1000:2000,0],spike_projected_u[1000:2000,2],'b*')
ax.plot(spike_projected_u[8000:9000,0],spike_projected_u[8000:9000,2],'k*')
plt.show()