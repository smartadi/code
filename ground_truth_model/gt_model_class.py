import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete, lti, dlti, dstep
from scipy import signal
from scipy.integrate import odeint
from scipy.stats import ortho_group

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

    def __init__(self,N: int,d: int):
        self.N = N
        self._d = d
        self.dt = 1/35  # experiment frequency matches sim frequency in discrete time
        self.t = np.linspace(0,self.dt*(self.N-1),self.N)
        

        #self.US = np.load(path + '/blue/svdSpatialComponents.npy')

        
        # noise parameters

        # disturbance parameters
        self._a = np.random.uniform(0,1,self._d)
        self._phi = np.random.uniform(-np.pi/2,-np.pi/2,self._d)
        self._omega = np.random.uniform(0,1,self._d)   # omega = 2*pi*f


        self.D = self._eigs()
        self.blk = self.get_blk_diag()
        self.A = self._sys_mat()
        self.temp = self._sim()
    

    def _eigs(self):
        D = np.random.uniform(-0.1, 0, (int(self._d/2),1)) + 1.j * np.random.uniform(0, 0.1, (int(self._d/2),1))
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

        print(P)
        print(P.dot(P.T))
        
        # A = B@self.blk@np.linalg.inv(B)
        A = P@self.blk@np.linalg.inv(P)
        # print(A)
        return A
    
    def _sim(self):
        x0 = np.random.uniform(-0.5,0.5,(self._d,1)).T
        x0 = x0/np.linalg.norm(x0)
        x0 = x0.ravel()
        print(np.linalg.norm(x0))
        print(x0)
        def model(x,t):
            noise = np.random.normal(0,0.0000001,self._d)
            distb = self._a*np.sin(self._omega*t+self._phi)
             
            return (self.A@x + distb)

        
        t = np.linspace(0,1000,35*1000+1)
        y = odeint(model,x0,t)

        plt.figure(1)
        plt.plot(t,y,'r-',linewidth=1)
        plt.xlabel('Time')
        plt.ylabel('Response (y)')
        plt.show()

        return y
        
    
        
        
G = ground_truth(1000,10)

# plt.plot(G.D.real,G.D.imag,'bo')
# plt.show()

# E,V = np.linalg.eig(G.A)
# print(E)
# print(G.D)



