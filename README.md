# Kalman

simple kalman filter class

This code is entirely adapted from filterpy by Roger R. Labbe.

https://filterpy.readthedocs.io/en/latest/
https://github.com/rlabbe/filterpy/tree/master/filterpy

The code is basic and uses numba jitclass for performance.
I added a function to predict angles by wrapping angles properly
in the update step.

python -m benchmarks.test_against_filterpy