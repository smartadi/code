import numpy as np

wf_space_data = np.load('data/svdSpatialComponents.npy')               #  US 500
n=20;

I = np.empty((wf_space_data.shape[0]*wf_space_data.shape[1],0))
for i in range(n):
    wf = wf_space_data[:,:,i]
    I = np.append(I,wf.reshape(-1,1),axis=1
                  )

print(I.shape)
np.savetxt("data/WF_US.csv", I, delimiter=",")