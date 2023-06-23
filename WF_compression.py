
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit, cuda
from numba import jit, prange, float32
import cv2

wf_temp_data = np.load('data/svdTemporalComponents_corr.npy')          #  V^T
wf_space_data = np.load('data/svdSpatialComponents.npy')               #  US 500



a = wf_space_data
VT = wf_temp_data.T

US = a.reshape(-1, 2000)

@jit(target_backend='cuda')
def mm(A,B,C):
    s = float(0)
    for i in range(560*560):
        AA = A[i,:]
        #C[i,:] = np.dot(AA,B)
        c = np.dot(AA,B)
        C[i, :] = c
        s = s + c**2
        #C[i, :] = int(np.dot(A[i, :], B))
    s = np.sqrt(s)

    for i in range(560*560):
        C[i,:] = C[i,:]/s*10000
    return C


USr = US[:,:500]

c = VT[0,:]


N=100  # number of images
data = np.zeros((100*100,N),dtype='int32')


for i in range(5):
    C = np.zeros((560 * 560, 1))
    cc = mm(USr,VT[:,i],C)
    im = cc.astype(int)
    ims = im.reshape(560,560)
    imr = cv2.resize(ims, (100, 100),interpolation=cv2.INTER_LINEAR_EXACT)
    #img_normalized = cv2.normalize(imr, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    d = imr.flatten('F')
    data[:,i] = d.T
    print(i)



plt.imshow(ims)
plt.show()

plt.imshow(imr)
plt.show()
