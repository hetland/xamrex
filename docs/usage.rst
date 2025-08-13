Basic Usage
===========

xamrex provides a unified xarray backend that automatically handles both single plotfiles and multi-time series datasets with full support for multi-level AMR data.

Unified xarray Backend
----------------------

The primary interface is through xarray's ``open_dataset`` function using the ``amrex`` engine, which automatically detects the input type:

Single Plotfile
^^^^^^^^^^^^^^^^

.. code-block:: python

   import xarray as xr
   
   # Load a single AMR level as an xarray dataset
   ds = xr.open_dataset('plt_00000', engine='amrex', level=0)
   
   # Access data variables (lazy loaded)
   temperature = ds['temp']
   salinity = ds['salt']
   
   # Access coordinates
   x_coords = ds.coords['x']
   y_coords = ds.coords['y']
   z_coords = ds.coords['z']  # if 3D data
   
   # Dataset attributes contain metadata
   print(f"Level: {ds.attrs['level']}")
   print(f"Time: {ds.attrs['current_time']}")
   print(f"Refinement factor: {ds.attrs['refinement_factor']}")

Multi-Time Series
^^^^^^^^^^^^^^^^^

The same backend automatically handles multiple plotfiles and concatenates them along the time dimension:

.. code-block:: python

   # Method 1: Explicit file list
   plotfiles = ['plt_00000', 'plt_01000', 'plt_02000']
   ds = xr.open_dataset(plotfiles, engine='amrex', level=0)
   
   # Method 2: Directory auto-discovery
   ds = xr.open_dataset('simulation_output/', engine='amrex', level=0, pattern='plt_*')
   
   # Method 3: Custom patterns
   ds = xr.open_dataset('data/', engine='amrex', level=0, pattern='sim_run_*')
   
   print(f"Time series shape: {ds.dims}")
   print(f"Time range: {ds.ocean_time.min().item()} to {ds.ocean_time.max().item()}")

Multi-Level AMR Support
-----------------------

Load any AMR level with automatic handling of missing levels:

.. code-block:: python

   # Load different refinement levels
   ds_level0 = xr.open_dataset(plotfiles, engine='amrex', level=0)  # Base level
   ds_level1 = xr.open_dataset(plotfiles, engine='amrex', level=1)  # Refined level
   
   print(f"Level 0: {dict(ds_level0.sizes)}")  
   print(f"Level 1: {dict(ds_level1.sizes)}")  # Higher resolution in x,y
   
   # Missing levels automatically filled with NaN
   # No errors if some time steps don't have the requested level

Automatic Level Detection
^^^^^^^^^^^^^^^^^^^^^^^^^

The backend intelligently handles mixed-level time series:

.. code-block:: python

   # Example: Mixed-level time series
   # plt_00000: max_level = 0 (base only)
   # plt_01000: max_level = 1 (has refinement)  
   # plt_02000: max_level = 1 (has refinement)
   
   ds_level1 = xr.open_dataset(['plt_00000', 'plt_01000', 'plt_02000'], 
                              engine='amrex', level=1)
   
   # Result: Level 1 dataset with:
   # - Time step 0: All NaN (plt_00000 doesn't have level 1)
   # - Time step 1: Valid data where level 1 exists, NaN elsewhere
   # - Time step 2: Valid data where level 1 exists, NaN elsewhere

Multi-Time Utility Functions
----------------------------

xamrex provides comprehensive utility functions for time series analysis:

Primary Functions
^^^^^^^^^^^^^^^^^

.. code-block:: python

   import xamrex
   
   # Primary time series loading function
   ds = xamrex.open_amrex_time_series(plotfiles, level=0)
   
   # Find plotfiles in directory  
   files = xamrex.find_amrex_time_series("data/", pattern="plt_*")
   
   # Create time series from directory
   ds = xamrex.create_time_series_from_directory("data/", pattern="plt_*", level=0)
   
   # Validate file compatibility before loading
   validation = xamrex.validate_time_series_compatibility(plotfiles)
   print(f"Compatible: {validation['compatible']}")

Time Series Analysis
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Extract time slices
   early = xamrex.extract_time_slice(ds, time_range=(0, 1000))
   middle = xamrex.extract_time_slice(ds, time_indices=slice(5, 10))
   
   # Compute time statistics
   stats = xamrex.compute_time_statistics(
       ds, 
       variables=['temp', 'salt'],
       statistics=['mean', 'std', 'min', 'max']
   )
   print(f"Statistics: {list(stats.data_vars)}")

Single-Level Utilities (Legacy Support)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Multi-level access utilities
   levels = xamrex.open_amrex_levels("plt_00000", levels=[0, 1, 2])
   summary = xamrex.create_level_summary("plt_00000")
   
   # Level information
   max_level = xamrex.get_max_level("plt_00000") 
   available = xamrex.get_available_levels_from_file("plt_00000")
   
   # Load specific levels
   ds_level0 = xamrex.load_base_level("plt_00000")
   ds_level1 = xamrex.load_level("plt_00000", level=1)

Advanced Usage
--------------

Custom Time Dimension
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Use custom time dimension name
   ds = xr.open_dataset(
       plotfiles, 
       engine='amrex', 
       level=0,
       time_dimension_name='time'  # Instead of default 'ocean_time'
   )

Custom Spatial Dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Rename spatial coordinates
   ds = xr.open_dataset(
       plotfiles, 
       engine='amrex', 
       level=0,
       dimension_names={'x': 'longitude', 'y': 'latitude', 'z': 'depth'}
   )

Memory Management
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Drop variables to save memory
   ds = xr.open_dataset(
       plotfiles, 
       engine='amrex', 
       level=0,
       drop_variables=['salt', 'other_field']
   )
   
   # Work with large time series efficiently
   large_ds = xamrex.open_amrex_time_series("large_simulation/plt_*", level=0)
   subset = large_ds.isel(ocean_time=slice(0, 10))  # Lazy slicing
   computed = subset.compute()  # Load only subset into memory

Lazy Loading and Performance
----------------------------

xamrex uses dask for lazy loading, providing excellent performance for large datasets:

.. code-block:: python

   ds = xr.open_dataset(plotfiles, engine='amrex', level=0)
   
   # This creates a lazy dask array - no data loaded yet
   temp = ds['temp']
   print(f"Data type: {type(temp.data)}")  # dask.array.core.Array
   
   # Data is loaded when you compute or access it
   temp_values = temp.compute()  # Now data is loaded
   
   # Or access a subset to load only what you need
   subset = temp.isel(x=slice(0, 100), y=slice(0, 100)).compute()

Large Time Series
^^^^^^^^^^^^^^^^^

Handle massive time series efficiently:

.. code-block:: python

   # Handle hundreds of time steps efficiently
   all_files = xamrex.find_amrex_time_series("massive_simulation/", "plt_*")
   print(f"Found {len(all_files)} files")  # Could be 1000+ files
   
   # Still loads quickly (metadata only)
   ds = xamrex.open_amrex_time_series(all_files, level=0)
   
   # Extract just what you need
   recent = ds.isel(ocean_time=slice(-10, None))  # Last 10 time steps
   subset = recent.sel(x=slice(0.25, 0.75))      # Spatial subset
   computed = subset.compute()                    # Only then load data

Working with Coordinates
------------------------

xamrex automatically calculates proper coordinates for each refinement level:

.. code-block:: python

   level_0 = xr.open_dataset(plotfiles, engine='amrex', level=0)
   level_1 = xr.open_dataset(plotfiles, engine='amrex', level=1)
   
   # Coordinates are properly scaled for each level
   dx_level_0 = level_0.coords['x'][1] - level_0.coords['x'][0]
   dx_level_1 = level_1.coords['x'][1] - level_1.coords['x'][0]
   
   print(f"Level 0 grid spacing: {dx_level_0.values}")
   print(f"Level 1 grid spacing: {dx_level_1.values}")
   print(f"Refinement ratio: {dx_level_0.values / dx_level_1.values}")

Refinement Patterns
^^^^^^^^^^^^^^^^^^^^

Understand how AMR refinement affects spatial dimensions:

.. code-block:: python

   # Level 0: Base resolution
   ds_l0 = xr.open_dataset(files, engine='amrex', level=0)
   print(f"Level 0: {dict(ds_l0.sizes)}")  # {'ocean_time': 3, 'z': 16, 'y': 15, 'x': 42}
   
   # Level 1: Refined in x,y but not z  
   ds_l1 = xr.open_dataset(files, engine='amrex', level=1) 
   print(f"Level 1: {dict(ds_l1.sizes)}")  # {'ocean_time': 3, 'z': 16, 'y': 45, 'x': 126}

Data Structure Examples
-----------------------

Single Time Step
^^^^^^^^^^^^^^^^^

.. code-block:: python

   <xarray.Dataset>
   Dimensions:  (ocean_time: 1, z: 16, y: 15, x: 42)
   Coordinates:
     * ocean_time  (ocean_time) float64 0.0
     * z           (z) float64 0.03125 0.09375 ... 0.96875
     * y           (y) float64 0.03125 0.09375 ... 0.96875  
     * x           (x) float64 0.03125 0.09375 ... 0.96875
   Data variables:
       temp         (ocean_time, z, y, x) float64 dask.array<chunksize=(1, 16, 15, 42)>
       salt         (ocean_time, z, y, x) float64 dask.array<chunksize=(1, 16, 15, 42)>

Time Series
^^^^^^^^^^^

.. code-block:: python

   <xarray.Dataset>
   Dimensions:  (ocean_time: 5, z: 16, y: 15, x: 42)
   Coordinates:
     * ocean_time  (ocean_time) float64 0.0 1000.0 2000.0 3000.0 4000.0
     * z           (z) float64 0.03125 0.09375 ... 0.96875
     * y           (y) float64 0.03125 0.09375 ... 0.96875
     * x           (x) float64 0.03125 0.09375 ... 0.96875
   Data variables:
       temp         (ocean_time, z, y, x) float64 dask.array<chunksize=(1, 16, 15, 42)>
       salt         (ocean_time, z, y, x) float64 dask.array<chunksize=(1, 16, 15, 42)>
   Attributes:
       concatenated_files: 5
       time_range: 0.0 to 4000.0
       level: 0

Error Handling and Validation
-----------------------------

Comprehensive error handling and validation:

.. code-block:: python

   import xamrex
   
   # Validate compatibility before loading
   validation = xamrex.validate_time_series_compatibility(plotfiles, level=1)
   
   if validation['compatible']:
       ds = xamrex.open_amrex_time_series(plotfiles, level=1)
   else:
       print(f"Issues: {validation['issues']}")
       print(f"Available fields: {validation['fields']}")

Common Error Scenarios
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   try:
       ds = xr.open_dataset('nonexistent_file', engine='amrex', level=0)
   except FileNotFoundError:
       print("Plotfile not found")
   
   try:
       # Try to load a level that doesn't exist in any file
       ds = xr.open_dataset(plotfiles, engine='amrex', level=5)
   except ValueError as e:
       print(f"Level error: {e}")
   
   # Check available levels first
   max_level = xamrex.get_max_level('plt_00000')
   print(f"Maximum available level: {max_level}")

Migration from Previous Versions
--------------------------------

The new API is fully backward compatible:

.. code-block:: python

   # Old way (still works exactly the same)
   ds = xr.open_dataset("plt_00000", engine='amrex', level=0)
   
   # New capabilities (no code changes needed for single files)
   ds = xr.open_dataset(["plt_00000", "plt_01000"], engine='amrex', level=0)  # Now works!
   ds = xr.open_dataset("simulation_data/", engine='amrex', level=0)          # Now works!

Input Type Summary
------------------

The unified backend accepts these input types via ``xr.open_dataset()``:

1. **Single plotfile directory**: ``"plt_00000"``
2. **List of plotfile directories**: ``["plt_00000", "plt_01000", "plt_02000"]``
3. **Directory containing plotfiles**: ``"simulation_output/"`` (with ``pattern="plt_*"``)
4. **Custom pattern matching**: ``"data/"`` (with ``pattern="sim_run_*"``)

All methods support the same parameters:
- ``level``: AMR level to load
- ``time_dimension_name``: Custom time dimension name
- ``dimension_names``: Custom spatial dimension names  
- ``drop_variables``: Variables to exclude
- ``pattern``: Glob pattern for directory scanning (methods 3 & 4)
