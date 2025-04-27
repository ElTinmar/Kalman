import numpy as np
import time
from kalman import KalmanFilter as myKF
from filterpy.kalman import KalmanFilter as fpKF

DIM_X = 9 
DIM_Z = 3
NREP = 1000

def benchmark():

    z = np.random.randn(DIM_Z, 1).astype(np.float32)
    for kf in [fpKF(DIM_X, DIM_Z), myKF(DIM_X, DIM_Z)]:

        # warmup (compile for numba)
        kf.predict()
        kf.update(z)

        # measure
        start = time.perf_counter()
        for _ in range(NREP):
            kf.predict()
            kf.update(z)
        end = time.perf_counter()
        elapsed = end - start
        print(f"{NREP} predict+update cycles took {elapsed:.6f} seconds")

if __name__ == '__main__':
    benchmark()