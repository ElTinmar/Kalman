import numpy as np
import pytest
from kalman import KalmanFilter, angle_diff  

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
    assert np.isclose(kf_2x1.x[0, 0], 2.0, atol=1e-2)

def test_update_wrap_angle(kf_angle):
    z = np.array([[-np.pi + 0.1]], dtype=np.float32)  
    kf_angle.predict()
    kf_angle.update_wrap_angle(z.reshape(-1,1), 0)
    diff = angle_diff(kf_angle.x[0, 0], -np.pi + 0.1)
    assert abs(diff) < 1e-1