API Reference
=============

Backend Entrypoint
------------------

.. autoclass:: xamrex.backend.AMReXCGridEntrypoint
   :members:
   :undoc-members:
   :show-inheritance:

The xarray backend entrypoint registered as ``engine='amrex'``.

Parameters
----------

When using ``xr.open_dataset(..., engine='amrex')``, the following parameters are supported:

**filename_or_obj** : str, Path, or list
    Single plotfile directory or list of plotfile directories for time series

**level** : int, default 0
    AMR level to load (0 = base level, 1+ = refined levels)

**drop_variables** : str or list of str, optional
    Variable names to exclude from the dataset

Example
-------

.. code-block:: python

   import xarray as xr
   
   # Single file
   ds = xr.open_dataset('plt00000', engine='amrex', level=0)
   
   # Time series
   ds = xr.open_dataset(['plt00000', 'plt00100'], engine='amrex', level=0)
   
   # With options
   ds = xr.open_dataset(
       files,
       engine='amrex',
       level=1,
       drop_variables=['w_vel', 'AKt']
   )

Core Components
---------------

The backend is implemented using these main classes:

**AMReXCGridStore**
    Data store for C-grid AMReX plotfiles with staggered variables

**FABLoader**
    Lazy loader for FAB (Fortran Array Box) data using dask

**CGridCoordinateGenerator**
    Generates properly scaled coordinates for C-grid points and AMR levels

**GridDetector**
    Automatically detects grid types and dimensions from plotfile structure

These classes are used internally and don't need to be accessed directly.
