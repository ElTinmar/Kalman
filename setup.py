from distutils.core import setup

setup(
    name='kalman',
    python_requires='>=3.8',
    author='Martin Privat',
    version='0.1.0',
    packages=['kalman'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    description='simple kalman filter',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy", 
        "scipy",
        "numba",
    ]
)
