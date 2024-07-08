
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
from numba import jit, cuda,prange

@jit(target_backend='cuda')
def mm(A,B,C):
    s = float(0)
    for i in prange(560*560):
        AA = A[i,:]
        #C[i,:] = np.dot(AA,B)
        c = np.dot(AA,B)
        C[i, :] = c
        s = s + c**2
        #C[i, :] = int(np.dot(A[i, :], B))
    s = np.sqrt(s)

    for i in prange(560*560):
        C[i,:] = C[i,:]/s*10000
    return C

@jit(target_backend='cuda')
def pix(im,x,y,k):
    p=0
    for i in prange(k):
        for j in prange(k):
            p = p + im[x+i-int(k/2),y+j-int(k/2)]
    return p

pathAB32 = '/run/user/1001/gvfs/smb-share:server=sahale.biostr.washington.edu,share=data/Subjects/AB_0032/2024-03-14/1'
pathAL14 = '/run/user/1001/gvfs/smb-share:server=sahale.biostr.washington.edu,share=data/Subjects/AL_0014/2022-08-15/1'

path = pathAL14

#input = np.load(path + '/lightCommand.raw.npy')
#print(input.shape)

#t_stamps = np.load(path + '/lightCommand.timestamps_Timeline.npy')
#print(t_stamps)




temp  = np.load(path + '/corr/svdTemporalComponents_corr.npy').T
print("dynamics shape")
print(temp.shape)

spat  = np.load(path + '/blue/svdSpatialComponents.npy')
print("Spatial shape")
print(spat.shape)

face_proc = np.load(path + '/face_proc.npy', allow_pickle=True).item()
yrange = face_proc['rois'][0]['yrange_bin']
xrange = face_proc['rois'][0]['xrange_bin']
mean = face_proc['avgframe'][0]

motTemp = face_proc['motSVD'][1]
motSpat = face_proc['motMask_reshape'][1]

print(motSpat.shape)
print(motTemp.shape)

print(mean.shape)

print(xrange.shape)
print(yrange.shape)



# reconstruct image and pixelate
n=100
U=np.zeros((560*560,0),int)
for i in range(n):
    U=np.column_stack([U,spat[:,:,i].flatten()])


print(U.shape)

t = 1000
Vi = temp[:n,t]

C = np.zeros((560 * 560, 1))
pip install opencv-python
cc = mm(U,Vi,C)
# im = cc.astype(int)
ims = cc.reshape(560,560)


x = 205
y = 305

plt.imshow(ims)
plt.plot(x,y,'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


p=0
k=10
# for i in range(k):
#     for j in range(k):
#         p = p + ims[x+i-int(k/2),y+j-int(k/2)]
#         # print(ims[x+i-k/2,y+j-k/2])

p = pix(ims,x,y,k)



fig, ax = plt.subplots(figsize=(9, 6))
pos = ax.imshow(ims, cmap='Blues', interpolation='none')
fig.colorbar(pos, ax=ax)
plt.show()





# Pixel value for entire dataset:
N=500
T0=10000
pt = np.zeros((N,1))
for s in range(N):
    Vi = temp[:n,T0+s]

    # get image
    C = np.zeros((560 * 560, 1))

    cc = mm(U,Vi,C)
    # im = cc.astype(int)
    imsi = cc.reshape(560,560)

    # get pixels kernel average
    p = 0
    k = 10
    # for i in range(k):
    #     for j in range(k):
    #         p = p + imsi[x+i-int(k/2),y+j-int(k/2)]/(k**2)
    #         #print(ims[x+i-k/2,y+j-k/2])
    p = pix(imsi,x,y,k)
    print(s)

    pt[s] = p

print(pt)

plt.plot(pt)
plt.show()



f, axarr = plt.subplots(1,2)
axarr[0].imshow(ims)
axarr[1].imshow(imsi)
plt.show()


# # read input-output datasets
# # Vf = np.genfromtxt(path+'/input/fs_output.csv', dtype=float, delimiter=',')
# # inpf = np.genfromtxt(path+'/input/fs_input.csv', dtype=float, delimiter=',')
# # tf = np.genfromtxt(path+'/input/fs_time.csv', dtype=float, delimiter=',')

# # Vi = np.genfromtxt(path+'/input/imp_output.csv', dtype=float, delimiter=',')
# # inpi = np.genfromtxt(path+'/input/imp_input.csv', dtype=float, delimiter=',')
# # ti = np.genfromtxt(path+'/input/imp_time.csv', dtype=float, delimiter=',')

# ''' 
# Vf = np.load(path+'/input/fs_output.csv')
# inpf = np.load(path+'/input/fs_input.csv')
# tf = np.load(path+'/input/fs_time.csv')

# Vi = np.load(path+'/input/imp_output.csv')
# inpi = np.load(path+'/input/imp_input.csv')
# ti = np.load(path+'/input/imp_time.csv')
# '''
# # print(Vf.shape)
# # print(Vf[0,:])
# # print(inpf.shape)
# # print(tf.shape)


# plt.plot(tf[0,:],Vf[:5,:].T)
# plt.show()

# plt.plot(tf[0,:],inpf[0,:].T)
# plt.show()

# Vf_re = Vf.reshape((n,-1))

# Vi_re = Vi.reshape((n,-1))

# print(Vi.shape)
# print(Vi_re.shape)

# T = Vi_re.shape

# print(T[1])


# N = T[1]
# #N = 1
# pti = np.zeros((N,1))

# for s in range(N):
#     Vi = Vi_re[:,s]

#     # get image
#     C = np.zeros((560 * 560, 1))

#     cc = mm(U,Vi,C)
#     # im = cc.astype(int)
#     imsi = cc.reshape(560,560)

#     # get pixels kernel average
#     p = 0
#     k = 10
#     for i in range(k):
#         for j in range(k):
#             p = p + imsi[x+i-int(k/2),y+j-int(k/2)]/(k**2)
#             #print(ims[x+i-k/2,y+j-k/2])
    
#     #print(s)

#     pti[s] = p

# plt.plot(pti)
# plt.show()

# Pti = pti.reshape(-1,41)


# print(Pti.shape)

# plt.plot(ti[0,:],Pti.T)
# plt.show()


# Tf = Vf_re.shape

# print(Tf[1])


# N = Tf[1]
# #N = 1


# ptf = np.zeros((N,1))
# for s in range(N):
#     Vf = Vf_re[:,s]

#     # get image
#     C = np.zeros((560 * 560, 1))

#     cc = mm(U,Vf,C)
#     # im = cc.astype(int)
#     imsi = cc.reshape(560,560)

#     # get pixels kernel average
#     p = 0
#     k = 10
#     for i in range(k):
#         for j in range(k):
#             p = p + imsi[x+i-int(k/2),y+j-int(k/2)]/(k**2)
#             #print(ims[x+i-k/2,y+j-k/2])
    
#     print(s)

#     ptf[s] = p

# plt.plot(ptf)
# plt.show()



# Ptf = ptf.reshape(-1,1101)

# print(Ptf.shape)



# np.save(path + '/input/pixel_imp',Pti)
# np.save(path + '/input/pixel_freq',Ptf)

# plt.plot(ti[0,:],Ptf.T)
# plt.show()


# # plt.plot(ti[0,:],Ptf[0,:].T)
# # plt.show()


# plt.plot(ti[0,:],Ptf[0,:])
# plt.show()




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
ax.plot(motTempfn[T0:T0+N,:])
ax.plot(pt)
ax.set(xlabel='time',ylabel = 'wf')
#ax.vlines(x = laser_on[:res],ymin = -0.02, ymax = 0.02, color = 'k', label = 'input')
plt.show()
