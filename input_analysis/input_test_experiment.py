# current paths are the exact files given by Anna due to root access structure on lab PC,
# I'll rewrite the code to use direct server paths
import traces as t
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.linalg import norm

dt=0.0285
t = np.linspace(0,6000)

path = "/run/user/1001/gvfs/smb-share:server=steinmetzsuper1.biostr.washington.edu,share=data/Subjects/ZYE_0069/2023-08-17/5"
dir_list = os.listdir(path)
print("Files and directories in '", path, "' :")
print("read data")

temp  = np.load(path + '/corr/svdTemporalComponents_corr.npy').T
print("dynamics shape")
print(temp.shape)
spat  = np.load(path + '/blue/svdSpatialComponents.npy')
print("Spatial shape")
print(spat.shape)

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

print("data rank")
r = 10
print(r)


# plt.plot(cam_times_short)
# plt.show()

# plt.plot(laser_on[:100],'bo')
# plt.plot(laser_off[:100],'ro')
# plt.show()

# plt.plot(laserX,laserY,'bo')
# plt.show()

# plt.plot(power)
# plt.show()


# plt.plot(cam_times_short[:500],temp[:500,:5])
# plt.show()

# ss = temp.T@temp
# print(ss.shape)
# print(ss)

# plt.plot(n)
# plt.show()

tempn = temp[:r,:]/np.linalg.norm(temp[:r,:])

# plt.plot(cam_times_short[:100],tempn[:100,:5])
# plt.show()
l=5000

# res = next(x for x, val in enumerate(laser_on)
#                                   if val > cam_times_short[l])

res = next(x for x, val in enumerate(laser_off)
                                  if val > cam_times_short[l])


fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(cam_times_short[:l],tempn[:5,:l].T)
ax.set(xlabel='time',ylabel = 'wf',title="dynamics at laser times")
ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = +0.02, color = 'b', label = 'input')
plt.show()



print(cam_times_short[l])
print(laser_on[:l])

t = cam_times_short[0]
print(t)
dt = 0.0285

tt = np.linspace(t,dt*l,l)


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





fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(cam_times_short[:l],tempn[:5,:l].T)
ax.set(xlabel='time',ylabel = 'wf')
ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = +0.02, color = 'b', label = 'input')
plt.show()

# Energy 
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(cam_times_short[:l],temp[:5,:l].T)
ax.set(xlabel='time',ylabel = 'wf')
ax.vlines(x = laser_on[:res],ymin = -1e6, ymax = 1e6, color = 'k', label = 'input')
plt.show()



En = norm(tempn,2,axis=0)
#print(En)

E1 = norm(tempn,1,axis=0)
#print(En)

plt.plot(En[:l])
plt.show()

# Energy 
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(cam_times_short[:l],tempn[:5,:l].T)
ax.plot(cam_times_short[:l],En[:l])
ax.set(xlabel='time',ylabel = 'wf')
ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
plt.show()



# Energy 
fig, axs = plt.subplots(5,figsize=(9, 6))
for i in range(5):
    axs[i].plot(cam_times_short[:l],tempn[i,:l])
    axs[i].plot(cam_times_short[:l],En[:l])
    axs[i].vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
plt.show()




fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(cam_times_short[10000:11000],tempn[:5,10000:11000])
ax.set(xlabel='time',ylabel = 'wf')
#ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
plt.show()




## Baseline data
basic_temp  = np.load('../data/svdTemporalComponents_corr.npy')
print(basic_temp.shape)




basic_tempn = basic_temp[:r,:]/np.linalg.norm(basic_temp[:r,:])

plt.plot(cam_times_short[:l],basic_tempn[:l,:5])
plt.show()


Eb = norm(basic_tempn,2,axis=1)

Eb1 = norm(basic_tempn,1,axis=1)


plt.plot(cam_times_short[:l],Eb[:l])
plt.show()

plt.plot(cam_times_short[:l],Eb1[:l])
plt.show()