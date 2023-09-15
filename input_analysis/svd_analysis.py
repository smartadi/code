import numpy as np

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