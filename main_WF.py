
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit, cuda
from numba import jit, prange, float32
import cv2

# import WF data (pre processed)
wf_temp_data = np.load('data/svdTemporalComponents_corr.npy')          #  V^T
wf_space_data = np.load('data/svdSpatialComponents.npy')               #  US 500

print(np.shape(wf_temp_data))
print(wf_temp_data.dtype)
print(np.shape(wf_space_data))
print(wf_space_data.dtype)

#plt.plot(wf_space_data[:, :, 1])


#plt.imshow(wf_space_data[:, :,0])
#wf_space_data_temp = np.memmap('data/svdSpatialComponents.npy',dtype='float32', mode='r', shape=(560, 560, 2000))

a = wf_space_data
VT = wf_temp_data.T
#plt.show()
print(np.shape(VT))

#plt.imshow(a[:, :, 1000])
#plt.show()
#print(wf_space_data_temp[:, :, 1])
#print(wf_space_data[:, :, :])
#
#
#
US = a.reshape(-1, 2000)
#
#
# # print(b.itemsize)
# # B = b.astype(np.float16)
# # print(b.itemsize)
# #
# # #u, s, vh = np.linalg.svd(b)
# #
# # #print(u.shape)
# # #print(s.shape)
# # #print(vh.shape)
# #
#
#
# c = b[:, 0:500]
# # C = c.astype(np.float16)
# d = wf_temp_data[0:50,0:500]
#
#
#
# # D = d.astype(np.float16)
# # print(C.itemsize)
# # print(D.itemsize)
# # print(c.shape)
# # print(d.shape)
# e = np.matmul(c, d.T)
# print(e.shape)
# f = e.reshape(560,560,50)
#
# plt.imshow(f[:,:,0])
# plt.show()

# ims = []
# fig = plt.figure()
# for i in range(50):
#     im = plt.imshow(f[:,:,i], animated=True)
#     ims.append([im])
#
# ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
#                                 repeat_delay=1000)
#
# ani.save('dynamic_images.mp4')
#
# plt.show()
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


#X_long = np.empty((560*560,10000))

USr = US[:,:500]

c = VT[0,:]

print(USr.shape)
print(c.shape)

N=5
data = np.zeros((100*100,N),dtype='int32')
#data = np.empty((240*240,N),dtype='int')


#for i in range(10):
for i in range(N):
    C = np.zeros((560 * 560, 1))
    cc = mm(USr,VT[:,i],C)
    im = cc.astype(int)
    ims = im.reshape(560,560)
    imr = cv2.resize(ims, (100, 100),interpolation=cv2.INTER_LINEAR_EXACT)
    #img_normalized = cv2.normalize(imr, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    d = imr.flatten('F')
    data[:,i] = d.T
    print(i)



print(data.dtype)
#np.save('data/data_WF_resize_50k_n100', data)
print(data.min())
print(data.max())

#print(a)
#print(c)

#images = USr @ VT[:, 0:100]

plt.imshow(ims)
plt.show()

plt.imshow(imr)
plt.show()

