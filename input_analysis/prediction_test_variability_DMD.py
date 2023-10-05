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

from HankelDMD_Predictor import HankelDMD_Predictor


# exp3 no stimulus data
path = "/run/user/1001/gvfs/smb-share:server=steinmetzsuper1.biostr.washington.edu,share=data/Subjects/ZYE_0069/2023-10-03/1"
dir_list = os.listdir(path)
print("Files and directories in '", path, "' :")
print("read data")

temp  = np.load(path + '/corr/svdTemporalComponents_corr.npy').T

print("dynamics shape")
print(temp.shape)
spat  = np.load(path + '/blue/svdSpatialComponents.npy')


# cam_times = np.load(path + '/cameraFrameTimes.npy')

# print(cam_times.shape)
# cam_times_short = cam_times[::2]
# print("camera times")
# print(cam_times_short.shape)

print("data rank")
r = 10
print(r)

#dt = 1/35
# dt = 0.02857 # fixed
# t0 = cam_times_short[0]
# t = np.linspace(t0,t0 + dt*(len(cam_times_short)-1),len(cam_times_short))


# plt.plot(cam_times_short[:500],temp[:5,:500].T)
# plt.show()


# normalising only r modes hides features
#tempn = temp[:r,:]/np.linalg.norm(temp[:r,:])

'''
a = np.linalg.norm(temp[:,:],2,0)
print(len(a))
tempn = temp[:,:]/a
print(tempn.shape)
print(np.linalg.norm(tempn[:,0]))
print(np.linalg.norm(tempn[0,:]))

nt = 1000

train_size = int(15/dt)
test_size = int(5/dt)
train_id = tempn[:,nt-train_size:nt-1]
time_id = cam_times_short[nt-train_size:nt-1]

test_id = tempn[:,nt-1:nt+test_size]
pred_id = cam_times_short[nt-1:nt+test_size]
print(time_id[-1])


fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(time_id,train_id[:r,:].T)
ax.set(xlabel='time',ylabel = 'wf')
#ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
plt.show()

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(pred_id,test_id[:r,:].T)
ax.set(xlabel='time',ylabel = 'wf')
#ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
plt.show()

print(test_id[:r,0])
print(train_id[:r,-1])
print(time_id[-1])
print(pred_id[0])



# N = test_size
# Hdmd = HankelDMD_Predictor(400,10)
# n=50
# Hdmd.fit(train_id[:n,:].T)

# #R = Hdmd.predict(N, reconstruct=True)
# R = Hdmd.predict(N, reconstruct=False)


# print(train_id.shape)
# print(test_id.shape)
# print(R.shape)


# fig, ax = plt.subplots(figsize=(9, 6))
# ax.plot(time_id,train_id[:r,:].T)
# ax.plot(time_id,R[:train_size-1,:r])
# ax.set(xlabel='time',ylabel = 'wf')
# #ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
# plt.show()

# r=5
# fig, ax = plt.subplots(figsize=(9, 6))
# ax.plot(pred_id,test_id[:r,:].T)
# ax.plot(pred_id[:-1],R[:,:r])
# ax.set(xlabel='time',ylabel = 'wf')
# #ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
# plt.show()


# # print(train_id[:r,-2])
# # print(train_id[:r,-1])
# # print(test_id[:r,0])
# # print(R[0,:r])

 
print(nt)
train_size = int(15/dt)
test_size = int(2/dt)
train_id = tempn[:,nt-train_size:nt-1]
time_id = cam_times_short[nt-train_size:nt-1]

test_id = tempn[:,nt-1:nt+test_size]
pred_id = cam_times_short[nt-1:nt+test_size]


n = 0
k = 25
d = 450
n_ts = 500
hdmdp = HankelDMD_Predictor(d,k)
hdmdp.fit(tempn[:50,n*n_ts:(n+1)*n_ts].T)
rec = hdmdp.predict(70,True)
fig,ax = plt.subplots(2,sharex=True,sharey=True)
ax[0].plot(tempn[:5,:570].T)
ax[0].axvline(500)
ax[0].set_ylabel("True Data")
ax[1].plot(rec)
ax[1].axvline(500)
ax[1].set_ylabel("Reconstruction+Forecast")
rec = hdmdp.predict(70,False)
fig,ax = plt.subplots(2,sharex=True,sharey=True)
ax[0].plot(tempn[:5,499:569].T)
ax[0].set_ylabel("True Data")
ax[1].plot(rec[:,:5])
ax[1].set_ylabel("Forecast")

print("Fit initial condition: " + str(rec[0]))
print("True initial condition: " + str(tempn[:5,(n+1)*n_ts-1]))
plt.show()



print(tempn.shape)


err = np.zeros((9,10))


fig,ax = plt.subplots(10,sharex=True,sharey=True)
# ax[0].plot(tempn[:5,:570].T)
# ax[0].axvline(500)
# ax[0].set_ylabel("True Data")
# ax[1].plot(rec)
# ax[1].axvline(500)
# ax[1].set_ylabel("Reconstruction+Forecast")
# rec = hdmdp.predict(70,False)
# fig,ax = plt.subplots(2,sharex=True,sharey=True)
# ax[0].plot(tempn[:5,499:569].T)
# ax[0].set_ylabel("True Data")
# ax[1].plot(rec[:,:5])
# ax[1].set_ylabel("Forecast")




for j in range(1):
    for i in range(10):
        nt = int((laser_on[idx_arr[j,i]] - cam_times_short[0])/dt)
        train_id = tempn[:,nt-train_size:nt-1]
        time_id = cam_times_short[nt-train_size:nt-1]

        test_id = tempn[:,nt-1:nt+test_size]
        pred_id = cam_times_short[nt-1:nt+test_size]

        #print(nt)

        N = test_size
        Hdmd = HankelDMD_Predictor(400,10)
        n=50
        Hdmd.fit(train_id[:n,:].T)
        R = Hdmd.predict(N, reconstruct=False)[:,:10]
        # ax[0].plot(test_id)
        # ax[0].set_ylabel("True Data")
        ax[i].plot(R[:,:5])
        ax[i].set_ylabel("Forecast")
        



        err[j,i] = np.linalg.norm(R - test_id[:10,:].T,2)

plt.show()
# print(err.shape)
# print(err)


# plt.plot(np.array(err),'bo')
# plt.show()


fig, axs = plt.subplots(8)
axs[0].plot(err[0,:],'bo')
axs[1].plot(err[1,:],'bo')
axs[2].plot(err[2,:],'bo')
axs[3].plot(err[3,:],'bo')
axs[4].plot(err[4,:],'bo')
axs[5].plot(err[5,:],'bo')
axs[6].plot(err[6,:],'bo')
axs[7].plot(err[7,:],'bo')


print(nt)
train_size = int(15/dt)
train_size = 500
test_size = int(2/dt)
train_id = tempn[:,nt-train_size:nt]
time_id = cam_times_short[nt-train_size:nt]

test_id = tempn[:,nt:nt+test_size]
pred_id = cam_times_short[nt:nt+test_size]
print(time_id[0])
print(time_id[-1])
print(time_id.shape)
print(train_id.shape)
print(pred_id[0])
print(pred_id[-1])
print(pred_id.shape)
print(test_id.shape)

N = test_size
Hdmd = HankelDMD_Predictor(400,101)
n = 50
Hdmd.fit(train_id[:n,:].T)

#R = Hdmd.predict(N, reconstruct=True)
R = Hdmd.predict(N, reconstruct=False)


# print(train_id.shape)
# print(test_id.shape)
# 
# print(tempn.shape)
# print(test_id.shape)


print(R.shape)
print(R[0,:5])
# print(R[1,:])
# print(test_id[:5,1])
# print(test_id[:5,0])
# print(train_id[:5,-1])
print(tempn[:5,nt-1])

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(pred_id,test_id[:r,:].T)
ax.plot(pred_id,R[:,:5])
ax.set(xlabel='time',ylabel = 'wf')
#ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
plt.show()


# np.savetxt("tempn1.csv",tempn[:,:10000], delimiter=",")
# np.savetxt("tempn2.csv",tempn[:,10000:20000], delimiter=",")
# np.savetxt("tempn3.csv",tempn[:,20000:30000], delimiter=",")
# np.savetxt("tempn4.csv",tempn[:,30000:40000], delimiter=",")
# np.savetxt("tempn5.csv",tempn[:,40000:50000], delimiter=",")


print(np.linalg.norm(tempn[:,0]))

print(tempn.shape)


fig, axs = plt.subplots(6)
axs[0].plot(tempn[:5,:10000].T)
axs[1].plot(tempn[:5,:1000].T)
axs[2].plot(tempn[:5,10000:11000].T)
axs[3].plot(tempn[:5,20000:21000].T)
axs[4].plot(tempn[:5,30000:31000].T)
axs[5].plot(tempn[:5,40000:41000].T)
plt.show()
'''