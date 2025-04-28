import numpy as np
import pytest
from kalman import KalmanFilter, angle_diff  
from filterpy.kalman import KalmanFilter as FilterPyKalman

@pytest.fixture
def kf_2x1():
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = np.array([[1, 1],
                     [0, 1]], dtype=np.float32)
    kf.H = np.array([[1, 0]], dtype=np.float32)
    kf.x = np.array([[0],
                     [1]], dtype=np.float32)
    kf.P *= 1.0
    return kf

@pytest.fixture
def kf_angle():
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.F = np.array([[1]], dtype=np.float32)
    kf.H = np.array([[1]], dtype=np.float32)
    kf.x = np.array([[np.pi - 0.1]], dtype=np.float32)
    kf.P *= 1.0
    return kf

def test_predict_only(kf_2x1):
    kf_2x1.predict()
    expected_x = np.array([[1],
                           [1]], dtype=np.float32)
    np.testing.assert_allclose(kf_2x1.x, expected_x, rtol=1e-5)

def test_update_without_missing(kf_2x1):
    z = np.array([[2]], dtype=np.float32)
    kf_2x1.predict()
    kf_2x1.update(z)
    assert np.isclose(kf_2x1.x[0, 0], 1.75, atol=1e-2)

def test_update_wrap_angle(kf_angle):
    z = np.array([[-np.pi + 0.1]], dtype=np.float32)  
    kf_angle.predict()
    kf_angle.update_wrap_angle(z.reshape(-1,1), 0)
    diff = angle_diff(kf_angle.x[0, 0], -np.pi + 0.1)
    assert abs(diff) < 1e-1


def test_against_filterpy():
    np.random.seed(0)

    dim_x = 4
    dim_z = 2
    n_steps = 100

    # Create your Numba Kalman filter
    kf_numba = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
    kf_numba.F = np.array([[1, 1, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 1],
                           [0, 0, 0, 1]], dtype=np.float32)
    kf_numba.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]], dtype=np.float32)
    kf_numba.Q *= 0.01
    kf_numba.R *= 0.1

    # Create equivalent FilterPy Kalman filter
    kf_fp = FilterPyKalman(dim_x=dim_x, dim_z=dim_z)
    kf_fp.x = np.zeros((dim_x, 1))
    kf_fp.F = np.array(kf_numba.F, dtype=np.float32)
    kf_fp.H = np.array(kf_numba.H, dtype=np.float32)
    kf_fp.Q = np.array(kf_numba.Q, dtype=np.float32)
    kf_fp.R = np.array(kf_numba.R, dtype=np.float32)
    kf_fp.P = np.eye(dim_x, dtype=np.float32)

    # Simulate random measurements
    zs = [np.random.randn(dim_z, 1).astype(np.float32) for _ in range(n_steps)]

    for z in zs:
        # Predict step
        kf_numba.predict()
        kf_fp.predict()

        # Update step
        kf_numba.update(z)
        kf_fp.update(z)

        # Compare states and covariances
        assert np.allclose(kf_numba.x, kf_fp.x, atol=1e-5), "State x mismatch"
        assert np.allclose(kf_numba.P, kf_fp.P, atol=1e-5), "Covariance P mismatch"
