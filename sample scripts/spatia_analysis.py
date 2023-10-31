# This script reads and plays with data, delete in later iterations.
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#float32

# exp3 no stimulus data
path = "/run/user/1001/gvfs/smb-share:server=steinmetzsuper1.biostr.washington.edu,share=data/Subjects/ZYE_0069/2023-10-03/1"
dir_list = os.listdir(path)
print("Files and directories in '", path, "' :")
print("read data")



# spat  = np.load(path + '/blue/svdSpatialComponents.npy')


# temp  = np.load(path + '/corr/svdTemporalComponents_corr.npy').T
# a = np.linalg.norm(temp[:,:],2,0)
# print(len(a))
# tempn = temp[:,:]/a
# print(tempn.shape)
# print(spat.shape)


spato  = np.load(path + '/corr/svdSpatialComponents_ortho.npy')
tempo  = np.load(path + '/corr/svdTemporalComponents_ortho.npy').T
print(tempo.shape)
print(spato.shape)
ao = np.linalg.norm(tempo[:,:],2,1)
print(len(ao))
tempno = tempo[:,:].T/ ao.T
print(tempno.shape)
print(spato.shape)


'''
spatn = spat.transpose(2,0,1).reshape(560*560,-1)
print(spatn.shape)
m =np.mean(spatn,0)

# print(m)
print(m.shape)


plt.plot(m)
plt.show()



f, axarr = plt.subplots(4,4)
for i in range(16):
    a = axarr[i//4,i%4].imshow(spat[:,:,i+0])
    f.colorbar(a)

'''
face_proc = np.load(path + '/face_proc.npy', allow_pickle=True).item()

motion = face_proc['motion'][1]
motions = motion[::2]

motions = motion[::2]

r=10  

motTemp = face_proc['motSVD'][1]
motSpat = face_proc['motMask_reshape'][1]

motTempr = motTemp[::2,:r]


motTempf = motTemp[::2,:]


motion = face_proc['motion'][1]
motions = motion[::2]

motTemp = face_proc['motSVD'][1]
motSpat = face_proc['motMask_reshape'][1]
motTempn = motTempr/np.linalg.norm(motTempr,2,0)
motTempfn = motTempf/np.linalg.norm(motTempf,2,0)

# a = axarr[2,2].imshow(spat[:,:,0])

# f.subplots_adjust(right=0.85)
# f.colorbar(a)

print(np.linalg.norm(spatn[:,0]))

print(spatn[:,0].T@spatn[:,1].T)


pixel = [440,340]
ptraj = spatn[pixel[0]*pixel[1],:500]@temp[:,10000:11000]


fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(motions[10000:11000]/1e6)
plt.show()

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(ptraj)
plt.show()


fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(tempno[:5,10000:11000 ].T)
plt.show()


fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(tempno[:5,:10000 ].T)
plt.show()
