#!/usr/bin/env python
# coding: utf-8

# Need data to be zero mean and scaled???
# Stochastic algorithm 3
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn import datasets  # to retrieve the iris Dataset
import pandas as pd  # to load the dataframe
from sklearn.preprocessing import StandardScaler  # to standardize the features
from sklearn.decomposition import PCA  # to apply PCA
import seaborn as sns  # to plot the heat maps#float32
import control
from numba import jit, prange

from past.utils import old_div

# Checking path to access other files
try:
    from sippy import *
except ImportError:
    import sys, os

    sys.path.append(os.pardir)
    from sippy import *

import numpy as np
from sippy import functionset as fset
from sippy import functionsetSIM as fsetSIM
import matplotlib.pyplot as plt



# Load spike data
#WF_data = np.load('../data/data_WF_PCA_projections_small.npy')
#WF_data = np.load('../data/data_WF_KPCA_projections_small.npy')
WF_data = np.load('../data/data_WF_UMAP_projections_small.npy')

print(WF_data.shape)
p=20
WF_data_r = WF_data[:,:p].T

print(WF_data_r.shape)
# Hankel Matrices i>n and j -> \infty
@jit(target_backend='cuda')
def Hankel(data,h,w):
    p = data.shape[0]
    J = w # 500
    I = h  # double
    Y = np.zeros((2*I*p,J))

    for i in range(2*I):
        Y[i*(p):(i+1)*p,:] = data[:,i:J+i]
    
    #print(Y.shape)
    return Y

#p = WF_data_r.shape[0]
I = 200
J = 2000


Y = Hankel(WF_data_r,I,J)
Yp = Y[:I*p,:]
Yf = Y[I*p:,:]

Ypp = Y[:(I+1)*p,:]
yff = Y[(I+1)*p:,:]
#print(Yp.shape)
#print(Yf.shape)
#print(Ypp.shape)
#print(yff.shape)

@jit(target_backend='cuda')
def Project(A,B):
    return A@B.T@np.linalg.pinv(B@B.T)@B 

@jit(target_backend='cuda')
def do_svd(a):
    uu, ss, vvh = np.linalg.svd(a, full_matrices=False)
    return uu, ss, vvh


Oi = Project(Yf,Yp)
oi = Project(yff,Ypp)

#W1 = np.eye(p*I)
#W2 = np.eye(J)
#O = W1@Oi@W2
O = Oi
#print(Oi.shape)
#print(O.shape)

U, s, VT = do_svd(O)

#print(s.shape)
fig, ax = plt.subplots()
ax.plot(s)
ax.set(xlabel='rank', ylabel='S',
       title='Singular values')
plt.show()

# get r from svd
r = 30
U_r = U[:,:r]
S_r = np.diag(s[:r,])


Gi = U_r@sc.linalg.sqrtm(S_r)
gi = Gi[:-p,:]
Xhat = np.linalg.pinv(Gi)@Oi
XXhat = np.linalg.pinv(gi)@oi
Yi = Yf[:p,:]

A = XXhat@np.linalg.pinv(Xhat)
#print(A.shape)
C = Yi@np.linalg.pinv(Xhat)
#print(C.shape)


# residuals
rho_w = XXhat - A@Xhat
rho_v = Yi - C@Xhat
rho = np.concatenate((rho_w, rho_v), axis=0)
#print(rho.shape)
Cov = np.cov(rho)
#print(Cov.shape)


Q = Cov[:30,:30]
R = Cov[-20:,-20:]
SS = Cov[:30,-20:]
#print(SS.shape)



# orignal data projected
t = 500
time = np.linspace(1, t, t)
neurons = np.linspace(1, p, p)
T, N = np.meshgrid(time, neurons)
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(T, N, WF_data_r[:,0:t], cmap = plt.cm.cividis)
fig.colorbar(surf, shrink=0.5, aspect=8)
plt.show()

#print(Xhat[:,0])
#print(WF_data_r[0:p,0])
######## Forecast
tt = 5000
x = np.zeros([r,tt])
x[:,0] = Xhat[:,0]
y = np.zeros([p,tt])

for i in range(tt-1):
    x[:,i+1] = A@x[:,i]
    y[:,i] = C@x[:,i]


print(WF_data_r[0,:])
print(y[0,:])

print(WF_data_r[0:p,0].shape)
print(y[:,0].shape)

# Forecast cost
t = tt
time = np.linspace(1, t, t)
neurons = np.linspace(1, p, p)
T, N = np.meshgrid(time, neurons)
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(T, N, y, cmap = plt.cm.cividis)
fig.colorbar(surf, shrink=0.5, aspect=8)
plt.show()
    


# In[ ]:


# orignal data projected
t = tt
time = np.linspace(1, t, t)
neurons = np.linspace(1, p, p)
T, N = np.meshgrid(time, neurons)

fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')

surf = ax.plot_surface(T, N, WF_data_r[0:p,0:t], cmap = plt.cm.cividis)


fig.colorbar(surf, shrink=0.5, aspect=8)

plt.show()


# In[ ]:





# In[ ]:


from mpl_toolkits.mplot3d.axes3d import get_test_data
from matplotlib import cm

# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=plt.figaspect(0.5))

# =============
# First subplot
# =============
# set up the axes for the first plot
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_title('predicted data')
time = np.linspace(1, t, t)
neurons = np.linspace(1, p, p)
T, N = np.meshgrid(time, neurons)


#surf = ax.plot_surface(T, N, spike_data[:,0:1000], cmap = plt.cm.cividis)
surf = ax.plot_surface(T, N, y, cmap = plt.cm.cividis)

fig.colorbar(surf, shrink=0.5, aspect=8)

# ==============
# Second subplot
# ==============
# set up the axes for the second plot
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_title('original data')

# plot a 3D wireframe like in the example mplot3d/wire3d_demo
surf = ax.plot_surface(T, N, WF_data_r[0:p,0:t], cmap = plt.cm.cividis)
fig.colorbar(surf, shrink=0.5, aspect=8)

plt.show()


# In[ ]:



fig, ax = plt.subplots(10,2)

for i in range(10):
    ax[i,0].plot(time,y[i,:t-1].T,'b')
    ax[i,1].plot(time,WF_data_r[i,:t-1].T,'b')
plt.show()
