xamrex
======

An xarray backend for reading AMReX plotfiles with C-grid staggered variable support and multi-level AMR.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api

Overview
--------

xamrex provides a simple xarray backend for AMReX simulation data with:

* **C-grid Support** - Automatic handling of staggered variables (rho, u, v, w, psi points)
* **Multi-level AMR** - Proper coordinate scaling for refined levels
* **Time Series** - Load multiple plotfiles with automatic concatenation
* **Lazy Loading** - Efficient memory usage via dask arrays
* **xgcm Compatible** - Grid-aware operations with xgcm

Quick Start
-----------

.. code-block:: python

   import xarray as xr
   from glob import glob

   # Single plotfile
   ds = xr.open_dataset('plt00000', engine='amrex', level=0)
   
   # Time series
   files = sorted(glob('ocean_out/plt*'))
   ds = xr.open_dataset(files, engine='amrex', level=0)
   
   # Refined level with proper coordinate scaling
   ds_level1 = xr.open_dataset(files, engine='amrex', level=1)
   
   # Access C-grid variables
   temp = ds['temp']      # rho-points: (ocean_time, z_rho, y_rho, x_rho)
   u_vel = ds['u_vel']    # u-points:   (ocean_time, z_rho, y_u, x_u)
   v_vel = ds['v_vel']    # v-points:   (ocean_time, z_rho, y_v, x_v)

Installation
------------

.. code-block:: bash

   pip install -e .

Requirements: Python ≥3.8, xarray, numpy, dask, pandas

Indices
=======

* :ref:`genindex`
* :ref:`search`
