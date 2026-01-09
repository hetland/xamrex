"""
Setup file for xamrex package.
"""
from setuptools import setup, find_packages

setup(
    name='xamrex',
    version='2.0.0',
    description='Xarray backend for AMReX plotfiles with C-grid support and AMR',
    author='Rob Hetland',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'xarray',
        'dask',
    ],
    entry_points={
        'xarray.backends': [
            'amrex=xamrex.backend:AMReXCGridEntrypoint',
        ],
    },
    python_requires='>=3.8',
)
