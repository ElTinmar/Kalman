import numpy as np
from numba import njit, float32, int64
from numba.experimental import jitclass
from numba.core.errors import NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

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

    def predict(self):
        
        # predict x
        self.x = self.F @ self.x
        
        #predict P
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        # innovation
        y = z - self.H @ self.x
        
        # compute kalman gain
        PHT = self.P @ self.H.T
        S = self.H @ PHT + self.R
        K = np.linalg.solve(S, PHT.T).T

        # update x
        self.x = self.x + K @ y

        # update P
        I_KH = self._I - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T


    def update_wrap_angle(self, z, which) -> None:

        # innovation
        Hx = self.H @ self.x
        y = z - Hx
        y[which, 0] = angle_diff(z[which,0], Hx[which,0])
        
        # compute kalman gain
        PHT = self.P @ self.H.T
        S = self.H @ PHT + self.R
        K = np.linalg.solve(S, PHT.T).T

        # update x
        self.x = self.x + K @ y
        self.x[which, 0] = wrap_angle(self.x[which, 0])

        # update P
        I_KH = self._I - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
