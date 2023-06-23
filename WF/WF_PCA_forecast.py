import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
from sklearn.decomposition import PCA  # to apply PCA
import matplotlib.animation as animation


r=10
# Load spike data
WF_data = np.load('../data/data_WF_PCA_projections_small.npy')
WF_forecast =np.float32( np.loadtxt('/home/nimbus/Documents/aditya/neuro/overschee book/subfun/WF_PCA_forecast_10k.csv', delimiter=","))
U = np.load('../data/WF_PCA_map.npy')
U_r = U[:,:r]
print(WF_data.dtype)
print(WF_forecast.dtype)
print(WF_data.shape)
print(WF_forecast.shape)
print(U.dtype)
print(U.shape)
print(WF_data[:10,-1])
print(WF_forecast[:10,-1])

WF_data_r = WF_data[:r,:]
Pca_forecast = U_r@WF_forecast
Pca_data = U_r@WF_data_r

im_forecast = Pca_forecast[:,-1].reshape(100,100)
im_data = Pca_data[:,-1].reshape(100,100)


plt.imshow(im_forecast)
plt.show()
plt.imshow(im_data)
plt.show()

ims=[]
fig= plt.figure()
for i in range(500):
    f = Pca_forecast[:, i].reshape(100, 100)
    #g = Pca_data[:, i].reshape(100, 100)
    im = plt.imshow(f, animated=True)
    #im2 = ax2.imshow(g, animated=True)
    
ims.append([im])

ani = animation.ArtistAnimation(fig,ims, interval=10, blit=True, repeat=False)

ani.save('PCA_forecast_images.mp4')

plt.show()