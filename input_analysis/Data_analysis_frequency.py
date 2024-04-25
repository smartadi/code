# current paths are the exact files given by Anna due to root access structure on lab PC,
# I'll rewrite the code to use direct server paths
import traces as t
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.linalg import norm
from scipy.interpolate import griddata
from scipy.interpolate import interpn
import sys
sys.path.append('/home/mist/Documents/projects/Brain/code/Matt/DMD')
import pandas
#from HankelDMD_Predictor import HankelDMD_Predictor


# exp3 no stimulus data

path = "/run/user/1001/gvfs/smb-share:server=steinmetzsuper1.biostr.washington.edu,share=data/Subjects/ZYE_0069/2023-10-03/1"
dir_list = os.listdir(path)
print("Files and directories in '", path, "' :")
print("read data")

ortho_path = path+'/ortho/'

ortho_temp = np.load(path + '/ortho/svdTemporalComponents_ortho.npy')
ortho_spat  = np.load(path + '/ortho/svdSpatialComponents_ortho.npy')



cam_times = np.load(path + '/corr/svdTemporalComponents_corr.timestamps.npy')



print("ortho shape")

print(ortho_temp.shape)
ao = np.linalg.norm(ortho_temp[:,:],2,0)
print(ao.shape)

ortho_tempn = ortho_temp[:,:]/ao
print(ortho_tempn.shape)

face_proc = np.load(path + '/face_proc.npy', allow_pickle=True).item()
yrange = face_proc['rois'][0]['yrange_bin']
xrange = face_proc['rois'][0]['xrange_bin']
mean = face_proc['avgframe'][0]
motTemp = face_proc['motSVD'][1]
motSpat = face_proc['motMask_reshape'][1]

print("motor shape")
print(motTemp.shape)


r=10  
motTempr = (motTemp[::2,:r]).T
motTempf = (motTemp[::2,:]).T

motion = face_proc['motion'][1]
motions = motion[::2]

motTempn = motTempr/np.linalg.norm(motTempr,2,0)
motTempfn = motTempf/np.linalg.norm(motTempf,2,0)

'''

f = np.genfromtxt('/home/mist/Documents/projects/Brain/code/SubspaceID/examples/freq.csv', delimiter=',')
print(f.shape)
print(f.dtype)



plt.rcParams['axes.grid'] = True
T = 40000
d = 3000
fig, axs = plt.subplots(4,1,figsize=(9, 6))
fig.suptitle('Vertically stacked subplots')
axs[0].plot(ortho_tempn[:10,T:T+d].T)
axs[0].set(xlabel='time',ylabel = 'wf_ortho_normalized')
axs[1].plot(motions[T:T+d])
axs[1].set(xlabel='time',ylabel = 'motions')
axs[2].plot(motTempfn[T:T+d,0])
axs[2].set(xlabel='time',ylabel = 'motion_svd')
axs[3].plot(f.T)
axs[3].set(xlabel='time',ylabel = 'discrete time eig')
#plt.grid(axis = 'x')
plt.show()




a = np.genfromtxt('/home/mist/Documents/projects/Brain/code/SubspaceID/examples/freq_sliding.csv', delimiter=',')
print(a.shape)
print(a.dtype)


plt.rcParams['axes.grid'] = True
T = 40000
d = 1000
fig, axs = plt.subplots(3,1,figsize=(9, 6))
fig.suptitle('Vertically stacked subplots')
axs[0].plot(ortho_tempn[:10,T:T+d].T)
axs[0].set(xlabel='time',ylabel = 'wf_ortho_normalized')
axs[1].plot(motTempfn[T:T+d,0])
axs[1].set(xlabel='time',ylabel = 'motion_svd')
axs[2].plot(a.T)
axs[2].set(xlabel='time',ylabel = 'discrete time eig')
#plt.grid(axis = 'x')
plt.show()
'''

## frequency analysis of signal for parameter selection

A = np.fft.rfftn(ortho_temp[:10,1000:2000])



print(A.shape)


fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(A.T)
ax.set(xlabel='time',ylabel = 'wf')
#ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
plt.show()



B = np.fft.rfftn(motTempn[:10,1000:2000])



print(B.shape)


fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(B.T)
ax.set(xlabel='fz',ylabel = '')
#ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
plt.show()