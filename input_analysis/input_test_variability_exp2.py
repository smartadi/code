# current paths are the exact files given by Anna due to root access structure on lab PC,
# I'll rewrite the code to use direct server paths
import traces as t
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.linalg import norm
from scipy.interpolate import griddata
from scipy.interpolate import interpn


# exp2
path = "/run/user/1001/gvfs/smb-share:server=steinmetzsuper1.biostr.washington.edu,share=data/Subjects/ZYE_0069/2023-08-24/4"
dir_list = os.listdir(path)
print("Files and directories in '", path, "' :")
print("read data")

temp  = np.load(path + '/corr/svdTemporalComponents_corr.npy').T
print("dynamics shape")
print(temp.shape)
# spat  = np.load(path + '/blue/svdSpatialComponents.npy')
# print("Spatial shape")
# print(spat.shape)

cam_times = np.load(path + '/cameraFrameTimes.npy')

print(cam_times.shape)
cam_times_short = cam_times[::2]
print("camera times")
print(cam_times_short.shape)

print("laser info")
laser_on = np.load(path + '/laserOnTimes.npy')
print(laser_on.shape)
laser_off = np.load(path + '/laserOffTimes.npy')
print(laser_off.shape)


laserX = np.load(path + '/galvoXPositions.npy')
laserY = np.load(path + '/galvoYPositions.npy')
print(laserX.shape)
print(laserY.shape)
power = np.load(path + '/laserPowers.npy')


TlaserX = np.load(path + '/galvoXCommand.npy').T
TlaserY = np.load(path + '/galvoYCommand.npy').T

print("data rank")
r = 10
print(r)

#dt = 1/35
dt = 0.02857 # fixed
t0 = cam_times_short[0]
t = np.linspace(t0,t0 + dt*(len(cam_times_short)-1),len(cam_times_short))


# plt.plot(cam_times_short[:500],temp[:5,:500].T)
# plt.show()


# normalising only r modes hides features
tempn = temp[:r,:]/np.linalg.norm(temp[:r,:])

#
tempn = temp[:,:]/np.linalg.norm(temp[:,:])

# plt.plot(cam_times_short[:100],tempn[:5,:100].T)
# plt.show()

# l = 10000

# res_on = next(x for x, val in enumerate(laser_on)
#                                   if val > cam_times_short[l])

# res = next(x for x, val in enumerate(laser_off)
#                                   if val > cam_times_short[l])


# fig, ax = plt.subplots(figsize=(9, 6))
# ax.plot(cam_times_short[:l],tempn[:5,:l].T)
# ax.set(xlabel='time',ylabel = 'wf',title="dynamics at laser times")
# ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = +0.02, color = 'b', label = 'input')
# plt.show()

# t = cam_times_short[0]
# print(t)

# acc = cam_times_short-t


# fig, ax = plt.subplots(figsize=(9, 6))
# ax.plot(cam_times_short[:l],tempn[:5,:l].T)
# ax.set(xlabel='time',ylabel = 'wf')
# ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = +0.02, color = 'b', label = 'input')
# plt.show()

# # Energy 
# # fig, ax = plt.subplots(figsize=(9, 6))
# # ax.plot(cam_times_short[:l],temp[:5,:l].T)
# # ax.set(xlabel='time',ylabel = 'wf')
# # ax.vlines(x = laser_on[:res],ymin = -1e6, ymax = 1e6, color = 'k', label = 'input')
# # plt.show()



# En = norm(tempn,2,axis=0)
# #print(En)

# E1 = norm(tempn,1,axis=0)
# #print(En)

# plt.plot(cam_times_short[:l],En[:l])
# plt.plot(cam_times_short[:l],E1[:l])
# plt.show()

# # # Energy 
# fig, ax = plt.subplots(figsize=(9, 6))
# ax.plot(cam_times_short[:l],tempn[:5,:l].T)
# ax.plot(cam_times_short[:l],En[:l])
# ax.plot(cam_times_short[:l],E1[:l])
# ax.set(xlabel='time',ylabel = 'wf')
# ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
# plt.show()



# # # Energy 
# # fig, axs = plt.subplots(5,figsize=(9, 6))
# # for i in range(5):
# #     axs[i].plot(cam_times_short[:l],tempn[i,:l])
# #     axs[i].plot(cam_times_short[:l],En[:l])
# #     axs[i].vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
# # plt.show()









# # ## Baseline data
# basic_temp  = np.load('../data/svdTemporalComponents_corr.npy').T
# print(basic_temp.shape)
# basic_tempn = basic_temp[:r,:]/np.linalg.norm(basic_temp[:r,:])


# Eb = norm(basic_tempn,2,axis=0)
# Eb1 = norm(basic_tempn,1,axis=0)


# plt.plot(cam_times_short[:l],basic_tempn[:5,:l].T)
# plt.plot(cam_times_short[:l],Eb[:l])
# plt.plot(cam_times_short[:l],Eb1[:l])
# plt.show()





# # # data = griddata(cam_times_short[:100], tempn[:,:100], t[:100], method='cubic')
# # # print(data.shape)
# # points = np.repeat(cam_times_short[:100], r, axis=1)
# # print(t.shape)
# # print(points.shape)
# # xi = []
# # # datan = interpn(points.T, tempn[:100,:].T, t[:100], method='nearest')
# # datan = interpn(cam_times_short[:100].T, tempn[:10,:100].T, t[:100], method='nearest')
# # # data_inp = interp(laser_on, tempn[:10,:100].T, t[:100], method='nearest')
# # print(datan.shape)
# # print(t)


# # fig, ax = plt.subplots(figsize=(9, 6))
# # ax.plot(cam_times_short[:100],tempn[0,:100].T)
# # ax.plot(cam_times_short[:100],datan)
# # ax.set(xlabel='time',ylabel = 'wf')
# # #ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
# # plt.show()


# # fig, ax = plt.subplots(figsize=(9, 6))
# # ax.plot(cam_times_short[:100],datan)
# # ax.set(xlabel='time',ylabel = 'wf')
# # #ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
# # plt.show()


# # fig, ax = plt.subplots(figsize=(9, 6))
# # ax.plot(cam_times_short[:100],tempn[0,:100].T)
# # ax.set(xlabel='time',ylabel = 'wf')
# # #ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
# # plt.show()





# # print(datan.dtype)


# # print("error = ")
# # a = np.linalg.norm(tempn[:,:100] - datan[:100,:].T,2,0)
# # print(a)

# # fig, ax = plt.subplots(figsize=(9, 6))
# # ax.plot(cam_times_short[:100],a)
# # ax.set(xlabel='time',ylabel = 'wf')
# # #ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
# # plt.show()


# # print("error = ")
# # te = cam_times_short[:100] - t[:100]
# # print(te)

# # ig, ax = plt.subplots(figsize=(9, 6))
# # ax.plot(cam_times_short[:100],te)
# # ax.set(xlabel='time',ylabel = 'wf')
# # #ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
# # plt.show()



# # difference metric

# dtempn = np.diff(tempn[:r,:l])
# gtempn =npa = np.asarray(np.gradient(tempn[:r,:l]), dtype=np.float32) 


# print(dtempn.shape)
# print(gtempn.shape)


# # plt.plot(cam_times_short[10:l],dtempn[:5,9:l].T)
# plt.plot(cam_times_short[:l],gtempn[0,:5,:l].T)
# ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
# plt.show()


# fig, ax = plt.subplots(figsize=(9, 6))
# ax.plot(cam_times_short[:l],gtempn[0,:5,:l].T)
# ax.plot(cam_times_short[:l],En[:l])
# ax.plot(cam_times_short[:l],E1[:l])
# ax.set(xlabel='time',ylabel = 'wf')
# ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
# plt.show()




# plt.plot(cam_times_short[:l],En[:l],'r')
# plt.plot(cam_times_short[:l],E1[:l],'g')
# plt.plot(cam_times_short[:l],Eb[:l],'b')
# plt.plot(cam_times_short[:l],Eb1[:l],'k')
# plt.xlabel('time')
# plt.ylabel('energy')
# plt.show()



# fig, axs = plt.subplots(2,figsize=(9, 6))
# axs[0].plot(cam_times_short[:l],Eb[:l])
# axs[0].plot(cam_times_short[:l],En[:l])
# axs[1].plot(cam_times_short[:l],E1[:l])
# axs[1].plot(cam_times_short[:l],Eb1[:l])
# plt.show()




plt.plot(TlaserX,TlaserY,'bo')
plt.plot(laserX,TlaserY,'ro')
plt.xlabel("x")
plt.ylabel("y")
plt.title("input locations commanded")
plt.show()



#variability
# find points with similar interest
print("variability")
# position array
pos = np.concatenate((TlaserX,TlaserY),axis=1)
print(TlaserX.shape)

upos = np.unique(pos, axis=0)
print(pos.shape)
# position indices

idx = []
for i in range(len(upos)):
    #print(np.where(np.all(pos==upos[i,:],axis=1)))
    idx.append(np.where(np.all(pos==upos[i,:],axis=1)))
    
idx_arr = np.concatenate(idx,axis=0)
print(idx_arr)
print(upos)

print(laser_on[idx_arr[0,:]])

# group
print(laser_on[idx_arr[0,0]] - cam_times_short[0])
nt = int((laser_on[idx_arr[0,0]] - cam_times_short[0])/dt)
print(nt)
print(cam_times_short[int(nt)-2])
print(cam_times_short[int(nt)-1])
print(cam_times_short[int(nt)])
print(cam_times_short[int(nt)+1])



# nt = (laser_on[idx_arr[0,1]] - cam_times_short[0])/dt
# print(nt)
# print(cam_times_short[int(nt)-2])
# print(cam_times_short[int(nt)-1])
# print(cam_times_short[int(nt)])
# print(cam_times_short[int(nt)+1])

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
print(laser_on[[idx_arr[0,0]]])


