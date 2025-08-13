from setuptools import setup, find_packages

setup(
    name="xamrex",
    version="0.5.0",
    packages=find_packages(),
    install_requires=[
        "xarray>=2023.1.0",
        "numpy>=1.20",
        "dask[array]>=2021.1",
        "pandas>=1.5.0",
    ],
    entry_points={
        'xarray.backends': [
            'amrex = xamrex.backend:AMReXEntrypoint',
        ],
    },
    author="xamrex contributors",
    description="AMReX plotfile backend for xarray with lazy dask loading and multi-time support.",
    long_description="A comprehensive xarray backend for reading AMReX plotfiles with lazy dask loading, supporting both single-level access and multi-time series concatenation following xarray conventions.",
    url="https://github.com/your-repo/xamrex",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    include_package_data=True,
)
