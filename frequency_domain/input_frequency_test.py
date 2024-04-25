# current paths are the exact files given by Anna due to root access structure on lab PC,
# I'll rewrite the code to use direct server paths
import traces as t
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.linalg import norm
from scipy.interpolate import griddata
from scipy.interpolate import interpn
from scipy import interpolate
from numba import jit, cuda

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

path = '/run/user/1001/gvfs/smb-share:server=sahale.biostr.washington.edu,share=data/Subjects/AB_0032/2024-03-14/1'
dir_list = os.listdir(path)
print("Files and directories in '", path, "' :")
print("read data")

input = np.load(path + '/lightCommand.raw.npy')
print(input.shape)

t_stamps = np.load(path + '/lightCommand.timestamps_Timeline.npy')
print(t_stamps)




# fig, ax = plt.subplots(figsize=(9, 6))
# ax.plot(input[2000000:2010000])
# ax.set(xlabel='time',ylabel = 'wf',title="dynamics at laser times")
# plt.show()



temp  = np.load(path + '/corr/svdTemporalComponents_corr.npy').T
print("dynamics shape")
print(temp.shape)

spat  = np.load(path + '/blue/svdSpatialComponents.npy')
print("Spatial shape")
print(spat.shape)
# tempo  = np.load(path + '/ortho/svdTemporalComponents_ZYE_ortho.npy')
# print("dynamics shape")
# print(temp.shape)

'''
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
# basic_temp  = np.load('../data/svdTemporalComponents_corr.npy').T
# print(basic_temp.shape)
# basic_tempn = basic_temp[:r,:]/np.linalg.norm(basic_temp[:r,:])


# Eb = norm(basic_tempn,2,axis=0)
# Eb1 = norm(basic_tempn,1,axis=0)


# plt.plot(cam_times_short[:l],basic_tempn[:5,:l].T)
# plt.plot(cam_times_short[:l],Eb[:l])
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
'''
T  = t_stamps[1,1].astype(int)
N  = t_stamps[1,0].astype(int)
t = np.arange(0,T,N)



a = T/1e5
print(a)
print(a.dtype)

b = a.astype(np.int64)
print(b)
print(b.dtype)


# laserX = np.load(path + '/galvoXPositions.npy')
# laserY = np.load(path + '/galvoYPositions.npy')
# print(laserX.shape)
# print(laserY.shape)





# reconstruct
n=10
U=np.zeros((560*560,0),int)
for i in range(n):
    U=np.column_stack([U,spat[:,:,i].flatten()])


print(U.shape)

t = 1000
Vi = temp[:n,t]

C = np.zeros((560 * 560, 1))

cc = mm(U,Vi,C)
# im = cc.astype(int)
ims = cc.reshape(560,560)


x = 205
y = 305

# plt.imshow(ims)
# plt.plot(x,y,'ro')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()


p=0
k=10
for i in range(k):
    for j in range(k):
        p = p + ims[x+i-int(k/2),y+j-int(k/2)]
        # print(ims[x+i-k/2,y+j-k/2])




fig, ax = plt.subplots(figsize=(9, 6))
pos = ax.imshow(ims, cmap='Blues', interpolation='none')
fig.colorbar(pos, ax=ax)
plt.show()





# Pixel value for entire dataset:
N=1
pt = np.zeros((N,1))
for s in range(N):
    Vi = temp[:n,50000+s]

    # get image
    C = np.zeros((560 * 560, 1))

    cc = mm(U,Vi,C)
    # im = cc.astype(int)
    imsi = cc.reshape(560,560)

    # get pixels kernel average
    p = 0
    k = 10
    for i in range(k):
        for j in range(k):
            p = p + imsi[x+i-int(k/2),y+j-int(k/2)]/(k**2)
            #print(ims[x+i-k/2,y+j-k/2])
    
    print(s)

    pt[s] = p

print(pt)

plt.plot(pt)
plt.show()



f, axarr = plt.subplots(1,2)
axarr[0].imshow(ims)
axarr[1].imshow(imsi)
plt.show()


# read input-output datasets

Vf = np.genfromtxt(path+'/input/fs_output.csv', dtype=float, delimiter=',')
inpf = np.genfromtxt(path+'/input/fs_input.csv', dtype=float, delimiter=',')
tf = np.genfromtxt(path+'/input/fs_time.csv', dtype=float, delimiter=',')

Vi = np.genfromtxt(path+'/input/imp_output.csv', dtype=float, delimiter=',')
inpi = np.genfromtxt(path+'/input/imp_input.csv', dtype=float, delimiter=',')
ti = np.genfromtxt(path+'/input/imp_time.csv', dtype=float, delimiter=',')

''' 
Vf = np.load(path+'/input/fs_output.csv')
inpf = np.load(path+'/input/fs_input.csv')
tf = np.load(path+'/input/fs_time.csv')

Vi = np.load(path+'/input/imp_output.csv')
inpi = np.load(path+'/input/imp_input.csv')
ti = np.load(path+'/input/imp_time.csv')
'''
print(Vf.shape)
print(Vf[0,:])
# print(inpf.shape)
# print(tf.shape)


plt.plot(tf[0,:],Vf[:5,:].T)
plt.show()

plt.plot(tf[0,:],inpf[0,:].T)
plt.show()

Vf_re = Vf.reshape((n,-1))

Vi_re = Vi.reshape((n,-1))

print(Vi.shape)
print(Vi_re.shape)

T = Vi_re.shape

print(T[1])


N = T[1]
#N = 1
pti = np.zeros((N,1))

for s in range(N):
    Vi = Vi_re[:,s]

    # get image
    C = np.zeros((560 * 560, 1))

    cc = mm(U,Vi,C)
    # im = cc.astype(int)
    imsi = cc.reshape(560,560)

    # get pixels kernel average
    p = 0
    k = 10
    for i in range(k):
        for j in range(k):
            p = p + imsi[x+i-int(k/2),y+j-int(k/2)]/(k**2)
            #print(ims[x+i-k/2,y+j-k/2])
    
    #print(s)

    pti[s] = p

plt.plot(pti)
plt.show()

Pti = pti.reshape(-1,41)


print(Pti.shape)

plt.plot(ti[0,:],Pti.T)
plt.show()


Tf = Vf_re.shape

print(Tf[1])


N = Tf[1]
#N = 1


ptf = np.zeros((N,1))
for s in range(N):
    Vf = Vf_re[:,s]

    # get image
    C = np.zeros((560 * 560, 1))

    cc = mm(U,Vf,C)
    # im = cc.astype(int)
    imsi = cc.reshape(560,560)

    # get pixels kernel average
    p = 0
    k = 10
    for i in range(k):
        for j in range(k):
            p = p + imsi[x+i-int(k/2),y+j-int(k/2)]/(k**2)
            #print(ims[x+i-k/2,y+j-k/2])
    
    print(s)

    ptf[s] = p

plt.plot(ptf)
plt.show()



Ptf = ptf.reshape(-1,1101)

print(Ptf.shape)



np.save(path + '/input/pixel_imp',Pti)
np.save(path + '/input/pixel_freq',Ptf)

plt.plot(ti[0,:],Ptf.T)
plt.show()


# plt.plot(ti[0,:],Ptf[0,:].T)
# plt.show()


plt.plot(ti[0,:],Ptf[0,:])
plt.show()


