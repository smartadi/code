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
tempo  = np.load(path + '/ortho/svdTemporalComponents_ZYE_ortho.npy')
print("dynamics shape")
print(temp.shape)


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



tempn = temp[:r,:]/np.linalg.norm(temp[:r,:])
tempon = tempo[:r,:]/np.linalg.norm(tempo[:r,:])


# plt.plot(cam_times_short[:100],tempn[:5,:100].T)
# plt.show()

l = 2500

l = 5000

res_on = next(x for x, val in enumerate(laser_on)
                                  if val > cam_times_short[l])

res = next(x for x, val in enumerate(laser_off)
                                  if val > cam_times_short[l])


fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(cam_times_short[:l],tempn[:5,:l].T)
ax.set(xlabel='time',ylabel = 'wf',title="dynamics at laser times")
ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = +0.02, color = 'b', label = 'input')
plt.show()

t = cam_times_short[0]
print(t)

acc = cam_times_short-t


fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(cam_times_short[:l],tempn[:5,:l].T)
ax.set(xlabel='time',ylabel = 'wf')
ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = +0.02, color = 'b', label = 'input')
plt.show()

# Energy 
# fig, ax = plt.subplots(figsize=(9, 6))
# ax.plot(cam_times_short[:l],temp[:5,:l].T)
# ax.set(xlabel='time',ylabel = 'wf')
# ax.vlines(x = laser_on[:res],ymin = -1e6, ymax = 1e6, color = 'k', label = 'input')
# plt.show()



En = norm(tempn,2,axis=0)
#print(En)

E1 = norm(tempn,1,axis=0)
E13 = norm(tempon,1,axis=0)

En3 = norm(tempon,2,axis=0)
#print(En)

plt.plot(cam_times_short[:l],En[:l])
plt.plot(cam_times_short[:l],E1[:l])
plt.show()

# # Energy 
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(cam_times_short[:l],tempn[:5,:l].T)
ax.plot(cam_times_short[:l],En[:l])
ax.plot(cam_times_short[:l],E1[:l])
ax.set(xlabel='time',ylabel = 'wf')
ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
plt.show()



# # Energy 
# fig, axs = plt.subplots(5,figsize=(9, 6))
# for i in range(5):
#     axs[i].plot(cam_times_short[:l],tempn[i,:l])
#     axs[i].plot(cam_times_short[:l],En[:l])
#     axs[i].vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
# plt.show()








'''
# ## Baseline data
basic_temp  = np.load('../data/svdTemporalComponents_corr.npy').T
print(basic_temp.shape)
basic_tempn = basic_temp[:r,:]/np.linalg.norm(basic_temp[:r,:])


Eb = norm(basic_tempn,2,axis=0)
Eb1 = norm(basic_tempn,1,axis=0)


plt.plot(cam_times_short[:l],basic_tempn[:5,:l].T)
plt.plot(cam_times_short[:l],Eb[:l])
plt.plot(cam_times_short[:l],Eb1[:l])
plt.show()





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



# difference metric

dtempn = np.diff(tempn[:r,:l])
gtempn =npa = np.asarray(np.gradient(tempn[:r,:l]), dtype=np.float32) 


print(dtempn.shape)
print(gtempn.shape)


# plt.plot(cam_times_short[10:l],dtempn[:5,9:l].T)
plt.plot(cam_times_short[:l],gtempn[0,:5,:l].T)
ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
plt.show()


fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(cam_times_short[:l],gtempn[0,:5,:l].T)
ax.plot(cam_times_short[:l],En[:l])
ax.plot(cam_times_short[:l],E1[:l])
ax.set(xlabel='time',ylabel = 'wf')
ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
plt.show()




plt.plot(cam_times_short[:l],En[:l],'r')
plt.plot(cam_times_short[:l],E1[:l],'g')
plt.plot(cam_times_short[:l],Eb[:l],'b')
plt.plot(cam_times_short[:l],Eb1[:l],'k')
plt.xlabel('time')
plt.ylabel('energy')
plt.show()



fig, axs = plt.subplots(2,figsize=(9, 6))
axs[0].plot(cam_times_short[:l],Eb[:l])
axs[0].plot(cam_times_short[:l],En[:l])
axs[1].plot(cam_times_short[:l],E1[:l])
axs[1].plot(cam_times_short[:l],Eb1[:l])
plt.show()

'''
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(cam_times_short[:l],tempon[:5,:l].T)
ax.set(xlabel='time',ylabel = 'wf',title="dynamics at laser times with orthogonal")
ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = +0.02, color = 'b', label = 'input')
plt.show()


plt.plot(cam_times_short[:l],En3[:l])
plt.plot(cam_times_short[:l],E13[:l])
plt.show()








face_proc = np.load(path + '/face_proc.npy', allow_pickle=True).item()
yrange = face_proc['rois'][0]['yrange_bin']
xrange = face_proc['rois'][0]['xrange_bin']
mean = face_proc['avgframe'][0]

motTemp = face_proc['motSVD'][1]
motSpat = face_proc['motMask_reshape'][1]

# print(motSpat.shape)
# print(motTemp.shape)

# print(mean.shape)

# print(xrange.shape)
# print(yrange.shape)

r=10  
motTempr = motTemp[::2,:r]


motTempf = motTemp[::2,:]


motion = face_proc['motion'][1]
motions = motion[::2]


print(face_proc.keys())

print(motion)

# plt.plot(motions)
# plt.show

# plt.plot(motTempr)
# plt.show


motTempn = motTempr/np.linalg.norm(motTempr,2,0)
motTempfn = motTempf/np.linalg.norm(motTempf,2,0)

plt.plot(motTempr[:1000,:])
plt.show

plt.plot(motions[:1000])
plt.show


fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(motTempn[4000:5000,:])
ax.set(xlabel='time',ylabel = 'wf')
#ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
plt.show()

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(motions[4000:5000])
ax.set(xlabel='time',ylabel = 'wf')
#ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
plt.show()


fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(motTempfn[4000:5000,:])
ax.set(xlabel='time',ylabel = 'wf')
#ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
plt.show()





fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(cam_times_short[:l],tempon[:5,:l].T)
ax.set(xlabel='time',ylabel = 'wf',title="dynamics at laser times with orthogonal")
ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = +0.02, color = 'b', label = 'input')
plt.show()

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(cam_times_short[:l],motions[:l])
ax.set(xlabel='time',ylabel = 'wf',title="motion ebergy at laser times with orthogonal")
ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = +0.02, color = 'b', label = 'input')
plt.show()