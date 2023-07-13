import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# wf_space_data = np.load('../data/svdSpatialComponents.npy')               #  US 500


xp = np.loadtxt("gt_dynamics.csv", delimiter=",", dtype=float)
ywf = np.loadtxt("gt_WF_dynamics.csv", delimiter=",", dtype=float)
ynp = np.loadtxt("gt_NP_dynamics.csv", delimiter=",", dtype=float)
t = xp[:, 0]
n = xp.shape[1] - 1
print(n)
x0 = np.loadtxt("gt_init.csv", delimiter=",", dtype=float)
um = np.loadtxt("gt_eigvec.csv", delimiter=",", dtype=str)
# Make a function to convert csv string to python complex
print(um)
Um = np.zeros((n,n),dtype=complex)

for i in range(n):
    for j in range(n):
        s = um[i,j]
        s = s.replace('i', 'j')
        v = complex(s)
        Um[i, j] = v

print(Um)
dm = np.loadtxt("gt_eigval.csv", delimiter=",", dtype=str)
print(dm)
Dm = np.zeros((n,n),dtype=complex)

for i in range(n):
    for j in range(n):
        s = dm[i,j]
        s = s.replace('i', 'j')
        v = complex(s)
        Dm[i,j] = v

print(Dm)


print(xp.shape)
print(ywf.shape)
print(ynp.shape)

# Generate a random non-trivial quadratic program.
T = 1000
N = 100

Q = 0.1*np.eye(n)
R = 0.1*np.eye(n)
IN = np.eye(N)
QN = np.kron(IN, Q)
RN = np.kron(IN, R)

F = np.array([])
G = np.array([])
F = np.empty((0,n), float)
G = np.empty((0,n*N), float)
print(Um.shape)
print(Dm.shape)
print(Um @ Dm**0 @ Um.T)
for i in range(N):

    F = np.real(np.append(F, Um @ Dm**i @ Um.T, axis=0))
    #F = np.vstack((F, Um @ Dm**i @ Um.T))

    #print(F)
    cc = np.empty((n,0), float)
    #cc = np.array([])

    for j in range(N):
        if j <= i:
            c = np.real(Um @ (Dm**(i-j+1)) @ Um.T)
        else:
            c = np.zeros((n, n))

        cc = np.append(cc, c, axis=1)
        #print(cc.shape)
    G = np.append(G, cc, axis=0)

print(F.shape)
print(G.shape)



B = cp.Variable(n*N)
#prob = cp.Problem(cp.Minimize((F@x0 + G@B).T @ QN @ (F@x0 + G@B) + B.T@RN@B))
prob = cp.Problem(cp.Minimize((cp.quad_form((F@x0 + G@B), QN) + (cp.quad_form(B, RN)))))

prob.solve()

print(prob.value)
print(B.value)
X = F@x0 + G@B.value

tn = np.linspace(0, 1, 100)
x = np.reshape(X, (n, N))
b = np.reshape(B.value, (n, N))
fig = plt.figure()
plt.plot(x[:,:].T)
plt.xlabel("states")
plt.ylabel("time")
plt.show()

print(x[:,0])

fig = plt.figure()
plt.plot(tn,b[:,].T)
plt.xlabel("states")
plt.ylabel("time")
plt.show()