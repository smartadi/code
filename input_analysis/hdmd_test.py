import sys
sys.path.append('/home/mist/Documents/projects/Brain/code/Matt/DMD')

import HankelDMD_Predictor

import numpy as np
import matplotlib.pylab as plt
plt.rcParams['figure.dpi']=400


path = "/run/user/1001/gvfs/smb-share:server=steinmetzsuper1.biostr.washington.edu,share=data/Subjects/ZYE_0069/2023-08-24/4"
dir_list = os.listdir(path)
print("Files and directories in '", path, "' :")
print("read data")

#SV  = np.load(path + '/corr/svdTemporalComponents_corr.npy')[:30000,:500].T
SV  = np.load('../data/svdTemporalComponents_corr.npy')[:30000,:500].T
print("dynamics shape")


n = 0

k = 5
d = 400
n_ts = 501
r = 50
p = 70
hdmdp = HankelDMD_Predictor.HankelDMD_Predictor(d,k)
#SV = np.load('../Mouse Data/svdTemporalComponents_corr.npy')
SVn = SV[:,:10000].T/np.linalg.norm(SV[:,:n_ts])
hdmdp.fit(SVn[n*n_ts:(n+1)*n_ts,:r])
rec = hdmdp.predict(p,True)
fig,ax = plt.subplots(2,sharex=True,sharey=True)
ax[0].plot(SVn[:n_ts+p])
ax[0].axvline(n_ts)
ax[0].set_ylabel("True Data")
ax[1].plot(rec)
ax[1].axvline(n_ts)
ax[1].set_ylabel("Reconstruction+Forecast")
rec = hdmdp.predict(p,False)
fig,ax = plt.subplots(2,sharex=True,sharey=True)
ax[0].plot(SVn[n_ts-1:n_ts+p])
ax[0].set_ylabel("True Data")
ax[1].plot(rec)
ax[1].set_ylabel("Forecast")
plt.show()


print(rec)