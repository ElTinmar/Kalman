import math
import numpy as np
from scipy.linalg import block_diag
from .kalman import KalmanFilter

def kinematic_state_transition(order, dt):
    """
    create a state transition matrix of a given order for a given time
    step `dt`.
    """

    if not(order >= 0 and int(order) == order):
        raise ValueError("order must be an int >= 0")

    # hard code common cases for computational efficiency
    if order == 0:
        return np.array([[1.]], dtype=np.float32)
    if order == 1:
        return np.array([[1., dt],
                         [0., 1.]], dtype=np.float32)
    if order == 2:
        return np.array([[1., dt, 0.5*dt*dt],
                         [0., 1., dt],
                         [0., 0., 1.]], dtype=np.float32)

    # grind it out computationally....
    N = order + 1

    F = np.zeros((N, N), dtype=np.float32)
    # compute highest order row
    for n in range(N):
        F[0, n] = float(dt**n) / math.factorial(n)

    # copy with a shift to get lower order rows
    for j in range(1, N):
        F[j, j:] = F[0, 0:-j]

    return F


def kinematic_kf(dim, order, dt=1., dim_z=1, order_by_dim=True, kf=None):
    """
    Returns a KalmanFilter using newtonian kinematics of arbitrary order
    for any number of dimensions. For example, a constant velocity filter
    in 3D space would have order 1 dimension 3.


    Examples
    --------

    A constant velocity filter in 3D space with delta time = .2 seconds
    would be created with

    >>> kf = kinematic_kf(dim=3, order=1, dt=.2)
    >>> kf.F
    >>> array([[1. , 0.2, 0. , 0. , 0. , 0. ],
               [0. , 1. , 0. , 0. , 0. , 0. ],
               [0. , 0. , 1. , 0.2, 0. , 0. ],
               [0. , 0. , 0. , 1. , 0. , 0. ],
               [0. , 0. , 0. , 0. , 1. , 0.2],
               [0. , 0. , 0. , 0. , 0. , 1. ]])


    which will set the state `x` to be interpreted as

    [x, x', y, y', z, z'].T

    If you set `order_by_dim` to False, then `x` is ordered as

    [x y z x' y' z'].T

    As another example, a 2D constant jerk is created with

    >> kinematic_kf(2, 3)


    Assumes that the measurement z is position in each dimension. If this is not
    true you will have to alter the H matrix by hand.

    P, Q, R are all set to the Identity matrix.

    H is assigned assuming the measurement is position, one per dimension `dim`.


    >>> kf = kinematic_kf(2, 1, dt=3.0)
    >>> kf.F
    array([[1., 3., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 3.],
           [0., 0., 0., 1.]])

    Parameters
    ----------

    dim : int, >= 1
        number of dimensions (2D space would be dim=2)

    order : int, >= 0
        order of the filter. 2 would be a const acceleration model with
        a stat

    dim_z : int, default 1
        size of z vector *per* dimension `dim`. Normally should be 1

    dt : float, default 1.0
        Time step. Used to create the state transition matrix

    order_by_dim : bool, default=True
        Defines ordering of variables in the state vector. `True` orders
        by keeping all derivatives of each dimensions)

        [x x' x'' y y' y'']

        whereas `False` interleaves the dimensions

        [x y z x' y' z' x'' y'' z'']

    kf : kalman filter like object, optional, default None
        Provide your own pre-created filter. This lets you use classes other
        than KalmanFilter.
    """

    if dim < 1:
        raise ValueError("dim must be >= 1")
    if order < 0:
        raise ValueError("order must be >= 0")
    if dim_z < 1:
        raise ValueError("dim_z must be >= 1")

    dim_x = order + 1

    if kf is None:
        kf = KalmanFilter(dim_x=dim * dim_x, dim_z=dim_z)
    assert kf.dim_x == dim * dim_x
    assert kf.dim_z == dim_z

    F = kinematic_state_transition(order, dt)
    if order_by_dim:
        diag = [F] * dim
        kf.F = block_diag(*diag)
    else:
        kf.F.fill(0.0)
        for i, x in enumerate(F.ravel()):
            f = np.eye(dim) * x

            ix, iy = (i // dim_x) * dim, (i % dim_x) * dim
            kf.F[ix:ix+dim, iy:iy+dim] = f

    if order_by_dim:
        for i in range(dim):
            kf.H[i, i * dim_x] = 1.
    else:
        for i in range(dim):
            kf.H[i, i] = 1.
            
    return kf