xamrex
======

A simplified xarray backend for reading AMReX plotfiles with lazy dask loading.

xamrex provides an efficient way to work with AMReX simulation data in Python, integrating seamlessly with the xarray ecosystem for analysis and visualization.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api

Key Features
------------

* **xarray Integration**: Seamless integration with xarray for familiar data analysis workflows
* **Lazy Loading**: Efficient memory usage with dask-backed lazy loading
* **Single-Level Focus**: Simplified interface focusing on single AMR level access
* **Multi-Level Utilities**: Utility functions for exploring and working with multi-level AMR data
* **Performance**: Optimized for large datasets with minimal memory footprint

Quick Start
-----------

.. code-block:: python

   import xarray as xr
   import xamrex

   # Load AMReX plotfile as xarray dataset
   ds = xr.open_dataset('path/to/plotfile', engine='amrex', level=0)
   
   # Access data variables
   temperature = ds['temp']
   salinity = ds['salt']
   
   # Use utility functions for multi-level analysis
   levels = xamrex.open_amrex_levels('path/to/plotfile')
   max_level = xamrex.get_max_level('path/to/plotfile')

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
