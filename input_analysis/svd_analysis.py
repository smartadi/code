import numpy as np
import matplotlib.pyplot as plt

from scipy import linalg

rng = np.random.default_rng()

m, n = 100, 1000

a = rng.standard_normal((m, n))# + 1.j*rng.standard_normal((m, n))

U, s, Vh = linalg.svd(a)

U.shape,  s.shape, Vh.shape

r=10
Vhr = Vh[:r,:]
sr = s[:r]


Sr = np.diag(sr)
SVhr = Sr@Vhr
print(s)
print(SVhr.shape)

print(np.dot(SVhr[0,:],SVhr[0,:]))
print(SVhr[8,:]@SVhr[0,:].T)

print(Vhr@Vhr.T)


temp  = np.load('/home/mist/Documents/projects/Brain/SVD_data/AL14 SVD Ortho/svdTemporalComponents_ortho.npy').T
spat  = np.load('/home/mist/Documents/projects/Brain/SVD_data/AL14 SVD Ortho/svdSpatialComponents_ortho.npy').T

print(spat.shape)
print(temp.shape)


tempn = temp/np.linalg.norm(temp)
print(tempn.shape)



plt.plot(tempn[:1000,:10])
plt.show()