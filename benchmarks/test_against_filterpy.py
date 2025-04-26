import numpy as np
import time
from kalman import KalmanFilter as myKF
from filterpy.kalman import KalmanFilter as fpKF

def benchmark():
    z = np.random.randn(3, 1).astype(np.float32)
    for kf in [fpKF(9, 3), myKF(9, 3)]:

        # warmup
        for _ in range(100):
            kf.predict()
            kf.update(z)

        # measure
        start = time.perf_counter()
        for _ in range(1000):
            kf.predict()
            kf.update(z)
        end = time.perf_counter()
        elapsed = end - start
        print(f"1000 predict+update cycles took {elapsed:.6f} seconds")

if __name__ == '__main__':
    benchmark()