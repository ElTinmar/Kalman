import numpy as np
import pytest
from kalman import kinematic_state_transition, kinematic_kf
from kalman import KalmanFilter


def test_kinematic_state_transition_common_cases():
    dt = 0.1

    # Order 0
    F = kinematic_state_transition(0, dt)
    np.testing.assert_array_almost_equal(F, np.array([[1.]], dtype=np.float32))

    # Order 1
    F = kinematic_state_transition(1, dt)
    expected = np.array([[1., dt],
                         [0., 1.]], dtype=np.float32)
    np.testing.assert_array_almost_equal(F, expected)

    # Order 2
    F = kinematic_state_transition(2, dt)
    expected = np.array([[1., dt, 0.5 * dt * dt],
                         [0., 1., dt],
                         [0., 0., 1.]], dtype=np.float32)
    np.testing.assert_array_almost_equal(F, expected)


def test_kinematic_state_transition_invalid_order():
    with pytest.raises(ValueError):
        kinematic_state_transition(-1, 1.0)

    with pytest.raises(ValueError):
        kinematic_state_transition(2.5, 1.0)


def test_kinematic_kf_order_by_dim_true():
    dt = 0.2
    dim = 3
    order = 1

    kf = kinematic_kf(dim=dim, order=order, dt=dt, order_by_dim=True)

    assert isinstance(kf, KalmanFilter)
    assert kf.F.shape == (dim * (order + 1), dim * (order + 1))
    assert kf.H.shape == (1, dim * (order + 1))

    # Check that F is block diagonal
    block = kinematic_state_transition(order, dt)
    for i in range(dim):
        idx = slice(i * (order + 1), (i + 1) * (order + 1))
        np.testing.assert_array_almost_equal(kf.F[idx, idx], block)


def test_kinematic_kf_order_by_dim_false():
    dt = 0.5
    dim = 2
    order = 1

    kf = kinematic_kf(dim=dim, order=order, dt=dt, order_by_dim=False)

    assert isinstance(kf, KalmanFilter)
    assert kf.F.shape == (dim * (order + 1), dim * (order + 1))

    # Check that the structure is interleaved
    dim_x = order + 1
    F_base = kinematic_state_transition(order, dt)
    for i in range(dim_x):
        for j in range(dim_x):
            expected_block = np.eye(dim) * F_base[i, j]
            block = kf.F[i * dim:(i+1)*dim, j * dim:(j+1)*dim]
            np.testing.assert_array_almost_equal(block, expected_block)


def test_kinematic_kf_invalid_parameters():
    with pytest.raises(ValueError):
        kinematic_kf(dim=0, order=1)

    with pytest.raises(ValueError):
        kinematic_kf(dim=2, order=-1)

    with pytest.raises(ValueError):
        kinematic_kf(dim=2, order=1, dim_z=0)

def test_kinematic_kf_predict_update():

    # Create a simple 1D constant velocity model
    kf = kinematic_kf(dim=1, order=1, dt=1.0)

    # Set initial state
    kf.x = np.array([[0],
                     [1]], dtype=np.float32)  # Position 0, velocity 1
    kf.P *= 1.0  # Initial uncertainty
    kf.R *= 0.1  # Measurement noise
    kf.Q *= 0.01 # Process noise

    # Predict step
    kf.predict()
    expected_predicted_x = np.array([[1],
                                     [1]], dtype=np.float32)
    np.testing.assert_allclose(kf.x, expected_predicted_x, rtol=1e-5)

    # Update step with a measurement (position = 1.2)
    z = np.array([[1.2]], dtype=np.float32)
    kf.update(z)

    # After update, position should move slightly toward measurement 1.2
    assert 1.0 < kf.x[0, 0] < 1.2
    # Velocity should stay close to 1
    assert 0.9 < kf.x[1, 0] < 1.1
    