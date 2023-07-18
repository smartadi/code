# This script reads and plays with data, delete in later iterations.
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#float32

'''

#spike_data = np.load('data/spikes_v1_clean.npy')                        #

#wf_temp_data = np.load('data/svdTemporalComponents_corr.npy')           #

#wf_space_data = np.load('data/svdSpatialComponents.npy')

#print(spike_data.itemsize)

print(wf_temp_data.itemsize)
print(wf_space_data.itemsize)
#wf_space_data_temp = np.memmap('data/svdSpatialComponents.npy', mode='r', shape=(560, 560, 2000))
print(np.shape(spike_data))
# print(spike_data[:, 1])

print(np.shape(wf_temp_data))
# print(wf_temp_data[:, 1])

print(np.shape(wf_space_data))
# print(wf_space_data[:, 1])

# plt.plot(wf_space_data[:, :, 1])

#print(wf_space_data_temp[:, :, 1])
#
# plt.imshow(wf_space_data[:, :,0])
#
# #a = wf_space_data_temp
#
# plt.show()
#
#
# #plt.imshow(a[:, :, 1000])
# #plt.show()
# #print(wf_space_data_temp[:, :, 1])
# #print(wf_space_data[:, :, :])
#
#
#
# b = wf_space_data.reshape(-1, 2000)
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
print(spike_data[100:200,0:100].T)
# spike plots
# plt.title("spkies")
# #plt.xlabel("X axis")
# #plt.ylabel("Y axis")
# #plt.eventplot(spike_data[:,100000:110000])
# plt.plot(spike_data[100:200,0:100].T,'o',color='black')
# plt.show()





spike_data_small = spike_data[:,0:10000]

fig, ax = plt.subplots()
light_onset_time = 4
stimulus_duration = 10
spike_value = 1
# Loop to plot raster for each trial
for trial in range(len(spike_data_small)):
    spike_times = [i for i, x in enumerate(spike_data_small[trial]) if x >= 1]
    ax.vlines(spike_times, trial - 0.5, trial + 0.5)

# ax.set_xlim([0, len(spike_data_small)])
ax.set_xlim([0, 10000])
ax.set_xlabel('Time (ms)')

# specify tick marks and label label y axis
# ax.set_yticks(range(len(spike_data_small)))
ax.set_ylabel('Trial Number')

ax.set_title('Neuronal Spike Times')

# add shading for stimulus duration)
plt.show()



# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# x, y = np.random.rand(2, 100) * 4
# hist, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[0, 4], [0, 4]])
#
# # Construct arrays for the anchor positions of the 16 bars.
# xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
# xpos = xpos.ravel()
# ypos = ypos.ravel()
# zpos = 0
#
# # Construct arrays with the dimensions for the 16 bars.
# dx = dy = 0.5 * np.ones_like(zpos)
# dz = hist.ravel()
#
# ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
#
# plt.show()

u, s, vh = np.linalg.svd(spike_data, full_matrices=False)

print(u.shape)
print(s.shape)
print(vh.shape)
r = 10
recon_spike = u[:, :r] @ np.diag(s[:r]) @ vh[:r, :]




recon_spike_small = recon_spike[:,0:10000]

fig, (ax1, ax2) = plt.subplots(1,2)
light_onset_time = 4
stimulus_duration = 10
spike_value = 1
# Loop to plot raster for each trial
for trial in range(len(recon_spike_small)):
    spike_times1 = [i for i, x in enumerate(recon_spike_small[trial]) if x >= 1]
    ax1.vlines(spike_times1, trial - 0.5, trial + 0.5)

# ax.set_xlim([0, len(spike_data_small)])
ax1.set_xlim([0, 10000])
ax1.set_xlabel('Time (ms)')

# specify tick marks and label label y axis
# ax.set_yticks(range(len(spike_data_small)))
ax1.set_ylabel('Trial Number')

ax1.set_title('Neuronal Spike Times')

# add shading for stimulus duration)


for trial in range(len(spike_data_small)):
    spike_times2 = [i for i, x in enumerate(spike_data_small[trial]) if x >= 1]
    ax2.vlines(spike_times2, trial - 0.5, trial + 0.5)

# ax.set_xlim([0, len(spike_data_small)])
ax2.set_xlim([0, 10000])
ax2.set_xlabel('Time (ms)')

# specify tick marks and label label y axis
# ax.set_yticks(range(len(spike_data_small)))
ax2.set_ylabel('Trial Number')

ax2.set_title('Neuronal Spike Times')
plt.show()


'''
import idx2numpy
import cv2
from numba import jit, prange

'''
stm_idx = idx2numpy.convert_from_file('data/linTest_stim_idx')
print(stm_idx.dtype)
print(stm_idx.shape)
print(stm_idx)
'''
b_spat = np.load('../data/behav_spat.npy')
b_temp = np.load('../data/behav_temp.npy')
b_spat_reshape = np.load('../data/behav_spat_reshape.npy')

print(b_spat.dtype)
print(b_spat.shape)

print(b_spat_reshape.dtype)
print(b_spat_reshape.shape)

print(b_temp.dtype)
print(b_temp.shape)



@jit(target_backend='cuda')
def mmult(a, b):
    c = a @ b
    return c



behave = mmult(b_spat, b_temp.T)

print(behave.dtype)
print(behave.shape)

im = behave[:,0]
image = im.reshape(122,116)

plt.imshow(image)
plt.show()


print(im)
print(max(im))
print(min(im))

'''
ims = []
fig = plt.figure()
for i in range(10000):
    ima = behave[:, i]
    image = ima.reshape(122, 116)
    im = plt.imshow(image, animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=10, blit=False,
                                repeat=False)

ani.save('behave_images_short.mp4')
'''


import pickle5 as pickle
file1 = open('../data/natImg_stim_idx', 'rb')
file2 = open('../data/linTest_stim_idx', 'rb')

nat_idx = pickle.load(file1)
lin_idx = pickle.load(file2)

file1.close()
file2.close()
l = list(nat_idx.keys())
print(l)
print(nat_idx[l[0]])

k = list(lin_idx.keys())
print(k)
print(lin_idx[k[0]])
