Usage Guide
===========

Basic Usage
-----------

Single Plotfile
^^^^^^^^^^^^^^^

Load a single AMReX plotfile at a specific level:

.. code-block:: python

   import xarray as xr
   
   # Load level 0 (base resolution)
   ds = xr.open_dataset('plt00000', engine='amrex', level=0)
   
   # Access variables
   temperature = ds['temp']
   salinity = ds['salt']
   
   print(f"Variables: {list(ds.data_vars)}")
   print(f"Dimensions: {dict(ds.dims)}")

Time Series
^^^^^^^^^^^

Load multiple plotfiles as a time series:

.. code-block:: python

   from glob import glob
   
   # Get sorted list of plotfiles
   files = sorted(glob('simulation/plt*'))
   
   # Load as time series
   ds = xr.open_dataset(files, engine='amrex', level=0)
   
   print(f"Time steps: {len(ds.ocean_time)}")
   print(f"Time range: {ds.ocean_time.values[0]} to {ds.ocean_time.values[-1]}")

C-Grid Staggered Variables
---------------------------

The backend automatically detects and handles C-grid staggered variables:

Variable Types
^^^^^^^^^^^^^^

.. code-block:: python

   ds = xr.open_dataset('plt00000', engine='amrex', level=0)
   
   # Rho-points (cell centers)
   temp = ds['temp']     # (ocean_time, z_rho, y_rho, x_rho)
   salt = ds['salt']
   
   # U-points (x-face centers)
   u_vel = ds['u_vel']   # (ocean_time, z_rho, y_u, x_u)
   
   # V-points (y-face centers)
   v_vel = ds['v_vel']   # (ocean_time, z_rho, y_v, x_v)
   
   # W-points (z-face centers)
   w_vel = ds['w_vel']   # (ocean_time, z_w, y_w, x_w)
   
   # 2D variables
   zeta = ds['zeta']     # (ocean_time, y_rho, x_rho)

xgcm Integration
^^^^^^^^^^^^^^^^

Use xgcm for grid-aware operations:

.. code-block:: python

   import xgcm
   
   # Create grid object
   grid = xgcm.Grid(ds, periodic=False)
   
   # Interpolate to u-points
   temp_at_u = grid.interp(ds.temp, 'X')
   
   # Calculate horizontal divergence
   div_h = grid.diff(ds.u_vel, 'X') + grid.diff(ds.v_vel, 'Y')

Multi-Level AMR
---------------

Loading Different Levels
^^^^^^^^^^^^^^^^^^^^^^^^^

Load any refinement level with proper coordinate scaling:

.. code-block:: python

   # Level 0: Base resolution
   ds0 = xr.open_dataset(files, engine='amrex', level=0)
   
   # Level 1: Refined (typically 2x or 3x finer)
   ds1 = xr.open_dataset(files, engine='amrex', level=1)
   
   print(f"Level 0: {dict(ds0.dims)}")
   print(f"Level 1: {dict(ds1.dims)}")
   
   # Coordinates are properly scaled
   dx0 = (ds0.x_rho[1] - ds0.x_rho[0]).values
   dx1 = (ds1.x_rho[1] - ds1.x_rho[0]).values
   print(f"Refinement ratio: {dx0 / dx1:.1f}x")

Level Masking
^^^^^^^^^^^^^

When a level doesn't exist at certain time steps, those times are filled with NaN:

.. code-block:: python

   # If some files don't have level 1
   # plt00000: only level 0
   # plt00100: levels 0 and 1
   # plt00200: levels 0 and 1
   
   ds_level1 = xr.open_dataset(['plt00000', 'plt00100', 'plt00200'], 
                               engine='amrex', level=1)
   
   # Time step 0: All NaN (level doesn't exist)
   # Time steps 1-2: Valid data where level 1 exists

Coordinates
^^^^^^^^^^^

Coordinates reflect actual physical locations at each level:

.. code-block:: python

   # Level 0 covers full domain with coarse resolution
   print(f"Level 0 x-range: {ds0.x_rho.min().values:.0f} to {ds0.x_rho.max().values:.0f}")
   
   # Level 1 covers refined patch with fine resolution
   print(f"Level 1 x-range: {ds1.x_rho.min().values:.0f} to {ds1.x_rho.max().values:.0f}")

Advanced Features
-----------------

Lazy Loading
^^^^^^^^^^^^

Data is loaded lazily using dask arrays:

.. code-block:: python

   ds = xr.open_dataset(files, engine='amrex', level=0)
   
   # No data loaded yet - just metadata
   temp = ds['temp']
   print(type(temp.data))  # dask.array.core.Array
   
   # Load only a subset
   subset = temp.isel(ocean_time=0, z_rho=0).compute()
   
   # Or select region before loading
   region = temp.sel(x_rho=slice(0, 50000)).compute()

Drop Variables
^^^^^^^^^^^^^^

Exclude variables to save memory:

.. code-block:: python

   ds = xr.open_dataset(
       files,
       engine='amrex',
       level=0,
       drop_variables=['w_vel', 'AKt']
   )

Working with Large Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Load metadata only (fast)
   ds = xr.open_dataset(files, engine='amrex', level=0)
   
   # Select subset before computing
   recent = ds.isel(ocean_time=slice(-10, None))  # Last 10 time steps
   surface = recent.sel(z_rho=0)                  # Surface only
   result = surface.compute()                     # Load only this subset

Common Patterns
---------------

Surface Temperature Time Series
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   files = sorted(glob('ocean_out/plt*'))
   ds = xr.open_dataset(files, engine='amrex', level=0)
   
   # Get surface temperature over time
   surf_temp = ds.temp.sel(z_rho=0)
   
   # Plot mean temperature
   import matplotlib.pyplot as plt
   surf_temp.mean(dim=['y_rho', 'x_rho']).plot()
   plt.ylabel('Temperature')
   plt.show()

Compare Refinement Levels
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   ds0 = xr.open_dataset(files, engine='amrex', level=0)
   ds1 = xr.open_dataset(files, engine='amrex', level=1)
   
   # Compare resolutions
   print(f"Level 0: {len(ds0.x_rho)} × {len(ds0.y_rho)} cells")
   print(f"Level 1: {len(ds1.x_rho)} × {len(ds1.y_rho)} cells")
   
   # Compare cell sizes
   dx0 = (ds0.x_rho[1] - ds0.x_rho[0]).values
   dx1 = (ds1.x_rho[1] - ds1.x_rho[0]).values
   print(f"Cell size ratio: {dx0/dx1:.1f}")

Extract Refined Region
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get data from refined patch
   ds1 = xr.open_dataset(files, engine='amrex', level=1)
   
   # Level 1 only exists in a subregion
   valid_data = ds1.temp.where(~ds1.temp.isnull())
   
   print(f"Refined region x: {ds1.x_rho.min().values:.0f} to {ds1.x_rho.max().values:.0f}")
   print(f"Refined region y: {ds1.y_rho.min().values:.0f} to {ds1.y_rho.max().values:.0f}")

Dataset Structure
-----------------

A typical dataset has this structure:

.. code-block:: python

   <xarray.Dataset>
   Dimensions:      (ocean_time: 2, z_rho: 32, y_rho: 48, x_rho: 38, ...)
   Coordinates:
     * ocean_time   (ocean_time) float64 0.0 1000.0
     * z_rho        (z_rho) float64 -19.69 -18.91 ... -0.3125
     * y_rho        (y_rho) float64 1042 3125 ... 98958
     * x_rho        (x_rho) float64 1053 3158 ... 78947
       y_u          (y_u) float64 1042 3125 ... 98958
       x_u          (x_u) float64 0.0 2105 ... 80000
       y_v          (y_v) float64 0.0 2083 ... 100000
       x_v          (x_v) float64 1053 3158 ... 78947
       z_w          (z_w) float64 -20.0 -19.38 ... 0.0
   Data variables:
       temp         (ocean_time, z_rho, y_rho, x_rho) float64 dask.array
       salt         (ocean_time, z_rho, y_rho, x_rho) float64 dask.array
       u_vel        (ocean_time, z_rho, y_u, x_u) float64 dask.array
       v_vel        (ocean_time, z_rho, y_v, x_v) float64 dask.array
       w_vel        (ocean_time, z_w, y_w, x_w) float64 dask.array
   Attributes:
       level:              0
       max_level_ever:     1
       dimensionality:     3
