# current paths are the exact files given by Anna due to root access structure on lab PC,
# I'll rewrite the code to use direct server paths
import traces as t
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.linalg import norm
from scipy.interpolate import griddata
from scipy.interpolate import interpn



path = "/run/user/1001/gvfs/smb-share:server=steinmetzsuper1.biostr.washington.edu,share=data/Subjects/ZYE_0069/2023-08-17/5"
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



plt.plot(laserX[:60],laserY[:60],'bo')
plt.plot(laserX[60:120],laserY[60:120],'ro')
plt.plot(laserX[120:180],laserY[120:180],'go')
plt.plot(laserX[180:240],laserY[180:240],'ko')
plt.plot(laserX[240:300],laserY[240:300],'mo')
plt.xlabel("x")
plt.ylabel("y")
plt.title("input locations actual")
plt.show()

# plt.plot(power)
# plt.show()


# plt.plot(cam_times_short[:500],temp[:500,:5])
# plt.show()

# ss = temp.T@temp
# print(ss.shape)
# print(ss)

# plt.plot(n)
# plt.show()

# tempn = temp[:r,:]/np.linalg.norm(temp[:r,:])

# # plt.plot(cam_times_short[:100],tempn[:100,:5])
# # plt.show()
# l = 5000

# # res = next(x for x, val in enumerate(laser_on)
# #                                   if val > cam_times_short[l])

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

# plt.plot(t,acc)
# plt.xlabel("time(sec)")
# plt.ylabel("lead error accumulation")
# plt.show()


# plt.plot(t[:99],cam_times_short[1:100] - cam_times_short[:99])
# plt.xlabel("time(sec)")
# plt.ylabel("dt")
# plt.show()

'''
dt = 0.0285

tt = np.linspace(t0,dt*l,l)


plt.plot(tt,tt,'b')
plt.plot(tt,cam_times_short[:l],'r')
plt.show()

print(tt)
print(cam_times_short[:l])




plt.plot(tt,cam_times_short[:l]-tt,'r')
plt.show()





diff_t = cam_times_short[1:] - cam_times_short[:-1]
dt = np.mean(diff_t)


ttt = np.absolute(diff_t - dt)
print(dt)

plt.plot(ttt)
plt.show()




'''
# fig, ax = plt.subplots(figsize=(9, 6))
# ax.plot(cam_times_short[:l],tempn[:5,:l].T)
# ax.set(xlabel='time',ylabel = 'wf')
# ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = +0.02, color = 'b', label = 'input')
# plt.show()

# # Energy 
# fig, ax = plt.subplots(figsize=(9, 6))
# ax.plot(cam_times_short[:l],temp[:5,:l].T)
# ax.set(xlabel='time',ylabel = 'wf')
# ax.vlines(x = laser_on[:res],ymin = -1e6, ymax = 1e6, color = 'k', label = 'input')
# plt.show()



# En = norm(tempn,2,axis=0)
# #print(En)

# E1 = norm(tempn,1,axis=0)
# #print(En)

# plt.plot(En[:l])
# plt.show()

# # Energy 
# fig, ax = plt.subplots(figsize=(9, 6))
# ax.plot(cam_times_short[:l],tempn[:5,:l].T)
# ax.plot(cam_times_short[:l],En[:l])
# ax.set(xlabel='time',ylabel = 'wf')
# ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
# plt.show()



# # Energy 
# fig, axs = plt.subplots(5,figsize=(9, 6))
# for i in range(5):
#     axs[i].plot(cam_times_short[:l],tempn[i,:l])
#     axs[i].plot(cam_times_short[:l],En[:l])
#     axs[i].vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
# plt.show()




# fig, ax = plt.subplots(figsize=(9, 6))
# ax.plot(cam_times_short[10000:11000],tempn[:5,10000:11000].T)
# ax.set(xlabel='time',ylabel = 'wf')
# #ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
# plt.show()




# ## Baseline data
# basic_temp  = np.load('../data/svdTemporalComponents_corr.npy').T
# print(basic_temp.shape)




# basic_tempn = basic_temp[:r,:]/np.linalg.norm(basic_temp[:,:r])

# plt.plot(cam_times_short[:l],basic_tempn[:5,:l].T)
# plt.show()


# Eb = norm(basic_tempn,2,axis=0)

# Eb1 = norm(basic_tempn,1,axis=0)


# plt.plot(cam_times_short[:l],Eb[:l])
# plt.show()

# plt.plot(cam_times_short[:l],Eb1[:l])
# plt.show()





# # data = griddata(cam_times_short[:100], tempn[:,:100], t[:100], method='cubic')
# # print(data.shape)
# points = np.repeat(cam_times_short[:100], r, axis=1)
# print(t.shape)
# print(points.shape)
# xi = []
# # datan = interpn(points.T, tempn[:100,:].T, t[:100], method='nearest')
# datan = interpn(cam_times_short[:100].T, tempn[:10,:100].T, t[:100], method='nearest')
# # data_inp = interp(laser_on, tempn[:10,:100].T, t[:100], method='nearest')
# print(datan.shape)
# print(t)


# fig, ax = plt.subplots(figsize=(9, 6))
# ax.plot(cam_times_short[:100],tempn[0,:100].T)
# ax.plot(cam_times_short[:100],datan)
# ax.set(xlabel='time',ylabel = 'wf')
# #ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
# plt.show()


# fig, ax = plt.subplots(figsize=(9, 6))
# ax.plot(cam_times_short[:100],datan)
# ax.set(xlabel='time',ylabel = 'wf')
# #ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
# plt.show()


# fig, ax = plt.subplots(figsize=(9, 6))
# ax.plot(cam_times_short[:100],tempn[0,:100].T)
# ax.set(xlabel='time',ylabel = 'wf')
# #ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
# plt.show()





# print(datan.dtype)


# print("error = ")
# a = np.linalg.norm(tempn[:,:100] - datan[:100,:].T,2,0)
# print(a)

# fig, ax = plt.subplots(figsize=(9, 6))
# ax.plot(cam_times_short[:100],a)
# ax.set(xlabel='time',ylabel = 'wf')
# #ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
# plt.show()


# print("error = ")
# te = cam_times_short[:100] - t[:100]
# print(te)

# ig, ax = plt.subplots(figsize=(9, 6))
# ax.plot(cam_times_short[:100],te)
# ax.set(xlabel='time',ylabel = 'wf')
# #ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
# plt.show()
print(TlaserX.shape)
print(laserX.shape)
print(laserX[0])
print(laserY[0])
print(laserX[1])
print(laserY[1])
print(laserX[2])
print(laserY[2])

print(TlaserX[0])
print(TlaserY[0])
print(TlaserX[1])
print(TlaserY[1])
print(TlaserX[2])
print(TlaserY[2])



# print(laserX[60])
# print(laserY[60])
# print(laserX[61])
# print(laserY[61])
# print(laserX[62])
# print(laserY[62])
# print(laserX[120])
# print(laserY[120])
# print(laserX[121])
# print(laserY[121])
# print(laserX[122])
# print(laserY[122])
# print(laserX[123])
# print(laserY[123])

# print(laserX[298])
# print(laserY[298])
# print(laserX[299])
# print(laserY[299])

# x = np.where(laserX == laserX[0])
# print(x) 

plt.plot(TlaserX[:60],TlaserY[:60],'bo')
plt.plot(TlaserX[60:120],TlaserY[60:120],'ro')
plt.plot(TlaserX[120:180],TlaserY[120:180],'go')
plt.plot(TlaserX[180:240],TlaserY[180:240],'ko')
plt.plot(TlaserX[240:300],TlaserY[240:300],'mo')
plt.show()



print(TlaserX[0])
print(TlaserY[0])
print(TlaserX[61])
print(TlaserY[61])
print(TlaserX[120])
print(TlaserY[120])
