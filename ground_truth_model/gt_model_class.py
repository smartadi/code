import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete, lti, dlti, dstep
from scipy import signal
from scipy.integrate import odeint
from scipy.stats import ortho_group
from scipy.signal import cont2discrete, lti, dlti, dstep

class ground_truth():
    """
    Class for generating a ground truth model.
    
    Generates a model with randomized parameters within certain bounds.

    Should always generate marginally stable system in no- noise case.

    Inputs:
    d: Number of delay embeddings used in forming Hankel matrix (int)
    svd_rank: Rank of SVD performed on Hankel matrix
    (Optional) dmd: DMD class used for fitting the Hankel matrix. Defaults to standard exact DMD (DMD class)
    Attributes:
    dmd: DMD class used for fitting. Contains modes, eigenvalues, amplitudes, etc. (DMD class)
    H: Hankel matrix form of the data (Numpy array)
    A: Low rank matrix that evolves the linear dynamics in the Hankel matrix (Numpy array)
    """

    def __init__(self,N: float,d: int):
        self.N = N
        self._d = d
        self.dt = 1/35  # experiment frequency matches sim frequency in discrete time
        #self.t = np.linspace(0,self.dt*(self.N-1),self.N)

        self.t = np.linspace(0,self.dt*(self.N-1),self.N)
        

        #self.US = np.load(path + '/blue/svdSpatialComponents.npy')

        
        # noise parameters

        # disturbance parameters
        self._a = np.random.uniform(0,1,self._d)
        self._phi = np.random.uniform(-np.pi/2,-np.pi/2,self._d)
        self._omega = np.random.uniform(0,1,self._d)   # omega = 2*pi*f


        self.D = self._eigs()
        self.blk = self.get_blk_diag()
        self.A, self.Ad = self._sys_mat()
        self.temp = self._sim()
    

    def _eigs(self):
        D = 0*np.random.uniform(-1, 0, (int(self._d/2),1)) + 10.j * np.random.uniform(0, 1, (int(self._d/2),1))
        D = np.vstack((D,np.conj(D)))
        #print(D.shape)
        #print(D)        
        return D
    
    def get_blk_diag(self):
        
        Dm = np.array([], dtype=np.int64).reshape(0,self._d)
        for i in range(int(self._d/2)):
            #dm = np.empty((2,2))
            #dm=[]
            dm = np.array([], dtype=np.int64).reshape(2,0)
            for j in range(int(self._d/2)):
                if i==j:
                    d = np.array([[np.real(self.D[i*2-1])[0],-np.imag(self.D[i*2-1])[0]],[np.imag(self.D[i*2-1])[0],np.real(self.D[i*2-1])[0]]]) 
                else:
                    d = np.zeros((2,2))
                # print(d)
                dm = np.hstack((dm,d))
            #print(dm)
            Dm = np.vstack((Dm,dm))
        return Dm
    
    def _sys_mat(self):
        B = np.random.uniform(-0.5,0.5,(self._d,self._d))

        P = ortho_group.rvs(dim=self._d)

        #print(P)
        #print(P.dot(P.T))
        
        # A = B@self.blk@np.linalg.inv(B)
        A = P@self.blk@np.linalg.inv(P)
        # print(A)


        # discretize
        
        B = np.zeros((self._d,1))
        C = np.zeros((1,self._d))
        D = np.zeros((1,1))
        l_system = lti(A, B, C, D)
        d_system = cont2discrete((A, B, C, D), self.dt, method='zoh')
        Ad = d_system[0]
        return A ,Ad

    
    def _sim(self):
        x0 = np.random.uniform(-1,1,(self._d,1)).T
        x0 = x0/np.linalg.norm(x0)
        x0 = x0.ravel()
        #print(np.linalg.norm(x0))
        #print(x0)
        def model(x,t):
            noise = 0*np.random.normal(0,1,self._d)
            distb = 0.1*self._a*np.sin(self._omega*t+self._phi)
             
            return (self.A@x + distb + noise)

        
        t = np.linspace(0,1000,35*1000+1)
        y = odeint(model,x0,self.t)

        plt.figure(1)
        plt.plot(self.t,y,'r-',linewidth=1)
        plt.xlabel('Time')
        plt.ylabel('Response (y)')
        plt.show()


        plt.figure(1)
        plt.plot(self.t[:500],y[:500,:],'r-',linewidth=1)
        plt.xlabel('Time')
        plt.ylabel('Response (y)')
        plt.show()

        #Discrete
        x = np.zeros((self._d,len(t)))
        x[:,1] = x0
        for i in range(len(t)-1):
            distb = 0.1*self._a*np.sin(self._omega*i*self.dt+self._phi)
            #print((self.Ad@x[:,i] + distb).type)
            x[:,i+1] = np.array(self.Ad@x[:,i] + distb)
            
        return x
        #return y
        
    
        
        
G = ground_truth(10000,10)

plt.plot(G.D.real,G.D.imag,'bo')
plt.show()

E,V = np.linalg.eig(G.A)
print(E)
# print(G.t)

noise = 0.1*np.random.normal(0,1,10)
print(noise.shape)
print(noise)

x0 = np.random.uniform(-1,1,(10,1)).T
x0 = x0/np.linalg.norm(x0)
x0 = x0.ravel()
t=1
print(x0)
phi = np.random.uniform(-np.pi/2,-np.pi/2,10)
omega = np.random.uniform(0,1,10)   # omega = 2*pi*f
distb = 0.1*np.random.uniform(0,1,10)*np.sin(omega*t+phi)
print(phi)
print(omega)
print(distb)

print(G.A@x0 + noise+distb)

print(G.A@x0)
print(G.Ad)