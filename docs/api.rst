API Reference
=============

This page contains the complete API reference for xamrex.

Unified Backend Entry Point
---------------------------

.. autoclass:: xamrex.AMReXEntrypoint
   :members:
   :undoc-members:

The unified xarray backend that handles all use cases automatically.

Multi-Time Series Functions
---------------------------

Primary Functions
^^^^^^^^^^^^^^^^^

.. autofunction:: xamrex.open_amrex_time_series

Primary function for loading time series datasets with automatic concatenation.

.. autofunction:: xamrex.find_amrex_time_series

Find and sort AMReX plotfiles in a directory by simulation time.

.. autofunction:: xamrex.create_time_series_from_directory

Create time series dataset from all matching plotfiles in a directory.

Validation and Analysis
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: xamrex.validate_time_series_compatibility

Validate that plotfiles can be concatenated into a time series.

.. autofunction:: xamrex.compute_time_statistics

Compute time-based statistics across the time series.

.. autofunction:: xamrex.extract_time_slice

Extract temporal subsets from time series datasets.

Single-Level Utility Functions
------------------------------

Multi-level Access
^^^^^^^^^^^^^^^^^^

.. autofunction:: xamrex.open_amrex_levels

Load multiple AMR levels from a single plotfile.

.. autofunction:: xamrex.get_available_levels_from_file

Get list of available AMR levels in a plotfile.

.. autofunction:: xamrex.get_max_level

Get the maximum AMR level available in a plotfile.

.. autofunction:: xamrex.load_level

Load a specific AMR level from a plotfile.

.. autofunction:: xamrex.load_base_level

Load the base level (level 0) from a plotfile.

Level Analysis
^^^^^^^^^^^^^^

.. autofunction:: xamrex.compare_level_resolutions

Compare grid spacing between different AMR levels.

.. autofunction:: xamrex.get_refinement_factors

Calculate refinement factors between AMR levels.

.. autofunction:: xamrex.extract_common_region

Extract the same spatial region from multiple AMR levels.

.. autofunction:: xamrex.create_level_summary

Create a summary of all available AMR levels.

.. autofunction:: xamrex.find_overlapping_region

Find spatial regions that overlap between AMR levels.

.. autofunction:: xamrex.calculate_effective_resolution

Calculate effective resolution for AMR level comparisons.

Core Backend Classes
--------------------

These classes implement the core functionality and are used internally:

Single-Level Store
^^^^^^^^^^^^^^^^^^

.. autoclass:: xamrex.backend.AMReXSingleLevelStore
   :members:
   :undoc-members:

Data store for a single AMR level from an AMReX plotfile.

Multi-Time Store
^^^^^^^^^^^^^^^^

.. autoclass:: xamrex.backend.AMReXMultiTimeStore
   :members:
   :undoc-members:

Data store for multiple AMReX plotfiles concatenated along time dimension.

Lazy Array Backend
^^^^^^^^^^^^^^^^^^

.. autoclass:: xamrex.backend.AMReXLazyArray
   :members:
   :undoc-members:

Lazy array wrapper for AMReX data using dask for memory efficiency.

Metadata Classes
----------------

These classes handle AMReX plotfile metadata parsing:

Dataset Metadata
^^^^^^^^^^^^^^^^

.. autoclass:: xamrex.AMReX_array.AMReXDatasetMeta
   :members:
   :undoc-members:

Metadata parser for AMReX plotfile headers and global information.

FAB Metadata
^^^^^^^^^^^^

.. autoclass:: xamrex.AMReX_array.AMReXFabsMetaSingleLevel
   :members:
   :undoc-members:

Metadata parser for FAB (Fortran Array Box) files at a single AMR level.

Function Parameters
-------------------

Common Parameters
^^^^^^^^^^^^^^^^^

Most functions in xamrex accept these common parameters:

**plotfile_path** : str or Path
    Path to AMReX plotfile directory

**level** : int, default 0
    AMR level to load (0 = base level)

**time_dimension_name** : str, optional
    Name for the time dimension (default: 'ocean_time')

**dimension_names** : dict, optional
    Custom spatial dimension names, e.g., {'x': 'longitude', 'y': 'latitude', 'z': 'depth'}

**drop_variables** : str or list of str, optional
    Variable names to exclude from the dataset

Multi-Time Parameters
^^^^^^^^^^^^^^^^^^^^

Functions dealing with time series accept additional parameters:

**plotfile_paths** : list of str or Path
    List of paths to AMReX plotfile directories

**pattern** : str, default "plt_*"
    Glob pattern to match plotfiles in directories

**sort_by_time** : bool, default True
    Whether to sort plotfiles by simulation time vs filename

Return Types
------------

Dataset Types
^^^^^^^^^^^^^

**xarray.Dataset**
    Standard xarray Dataset with lazy dask arrays for data variables and proper coordinates

**dict of xarray.Dataset**
    For multi-level functions, returns dict mapping level numbers to datasets

Validation Types
^^^^^^^^^^^^^^^^

**dict**
    Validation functions return dictionaries with keys:
    
    - ``compatible`` : bool - Whether files are compatible
    - ``issues`` : list - List of compatibility issues  
    - ``file_count`` : int - Number of files validated
    - ``time_range`` : tuple - (start_time, end_time)
    - ``fields`` : list - Available field names
    - ``domain_info`` : dict - Domain boundary information

Usage Examples
--------------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   import xarray as xr
   import xamrex
   
   # Single plotfile
   ds = xr.open_dataset('plt_00000', engine='amrex', level=0)
   
   # Time series  
   ds = xr.open_dataset(['plt_00000', 'plt_01000'], engine='amrex', level=0)
   
   # Directory auto-discovery
   ds = xr.open_dataset('data/', engine='amrex', level=0, pattern='plt_*')

Advanced Usage
^^^^^^^^^^^^^^

.. code-block:: python

   # Multi-level time series with custom parameters
   ds = xr.open_dataset(
       plotfiles,
       engine='amrex', 
       level=1,
       time_dimension_name='time',
       dimension_names={'x': 'lon', 'y': 'lat'},
       drop_variables=['salt']
   )
   
   # Validation before loading
   validation = xamrex.validate_time_series_compatibility(plotfiles, level=1)
   if validation['compatible']:
       ds = xamrex.open_amrex_time_series(plotfiles, level=1)
   
   # Time series analysis
   stats = xamrex.compute_time_statistics(ds, variables=['temp'], statistics=['mean'])
   subset = xamrex.extract_time_slice(ds, time_range=(1000, 5000))

Error Handling
--------------

Common Exceptions
^^^^^^^^^^^^^^^^^

**ValueError**
    Raised when:
    - Requested AMR level doesn't exist in any file
    - Plotfiles have incompatible domains or fields
    - Invalid parameter combinations

**FileNotFoundError**
    Raised when:
    - Plotfile directory doesn't exist
    - Required AMReX files (Header, Level_0, etc.) are missing

**KeyError**
    Raised when:
    - Requested field doesn't exist in plotfile
    - Invalid dimension names specified

Best Practices
--------------

Performance
^^^^^^^^^^^

- Use ``drop_variables`` to exclude unneeded fields and save memory
- Access subsets with ``.isel()`` or ``.sel()`` before calling ``.compute()``
- For large time series, process data in chunks rather than loading everything

Memory Management
^^^^^^^^^^^^^^^^^

- Take advantage of lazy loading - data isn't loaded until ``.compute()`` is called
- Use temporal and spatial slicing to work with manageable data sizes
- For very large datasets, consider using dask's distributed computing features

Validation
^^^^^^^^^^

- Always validate time series compatibility before loading large datasets
- Check available levels with ``get_max_level()`` before requesting specific levels
- Use ``find_amrex_time_series()`` to discover available plotfiles

Version Information
-------------------

This API reference covers xamrex version 0.5.0 and later, which includes:

- Unified backend supporting both single and multi-time datasets
- Multi-level AMR support with automatic NaN filling  
- Comprehensive time series analysis tools
- Full backward compatibility with previous versions
