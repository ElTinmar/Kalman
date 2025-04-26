import numpy as np
from numba import njit, float32, int64
from numba.experimental import jitclass

@njit(float32(float32))
def wrap_angle(a: float) -> float:
    return ((a + np.pi) % (2*np.pi)) - np.pi

@njit(float32(float32, float32))
def angle_diff(a1: float, a2: float) -> float: 
    return wrap_angle(a1 - a2)

spec = [
    ('dim_x', int64),
    ('dim_z', int64),
    ('x', float32[:, :]),
    ('P', float32[:, :]),
    ('Q', float32[:, :]),
    ('F', float32[:, :]),
    ('H', float32[:, :]),
    ('R', float32[:, :]),
    ('_I', float32[:, :]),
]

@jitclass(spec)
class KalmanFilter:

    def __init__(self, dim_x: int, dim_z: int) -> None:

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = np.zeros((dim_x, 1), dtype=np.float32)        
        self.P = np.eye(dim_x, dtype=np.float32)               
        self.Q = np.eye(dim_x, dtype=np.float32)               
        self.F = np.eye(dim_x, dtype=np.float32)               
        self.H = np.zeros((dim_z, dim_x), dtype=np.float32)    
        self.R = np.eye(dim_z, dtype=np.float32)               
        self._I = np.eye(dim_x, dtype=np.float32)

    def predict(self) -> None:
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z) -> None:
        y = z - np.dot(self.H, self.x)

        PHT = np.dot(self.P, self.H.T)
        S = np.dot(self.H, PHT) + self.R
        K = np.linalg.solve(S, PHT.T).T
        self.x = self.x + np.dot(K, y)

        I_KH = self._I - np.dot(K, self.H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(K, self.R), K.T)

    def update_wrap_angle(self, z, which) -> None:
        Hx = np.dot(self.H, self.x)
        y = z - Hx
        y[which, 0] = angle_diff(z[which,0], Hx[which,0])

        PHT = np.dot(self.P, self.H.T)
        S = np.dot(self.H, PHT) + self.R
        K = np.linalg.solve(S, PHT.T).T
        self.x = self.x + np.dot(K, y)
        self.x[which, 0] = wrap_angle(self.x[which, 0])

        I_KH = self._I - np.dot(K, self.H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(K, self.R), K.T)
