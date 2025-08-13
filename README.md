# xamrex: Unified AMReX Backend for xarray

A comprehensive xarray backend for reading AMReX plotfiles with native support for time series concatenation and multi-level AMR data.

## Overview

xamrex provides a unified interface for working with AMReX simulation data in Python, seamlessly integrating with the xarray ecosystem. The package supports both single plotfiles and multi-time series datasets, with intelligent handling of adaptive mesh refinement (AMR) levels.

## Key Features

- **Unified xarray Backend**: Single `amrex` engine handles all use cases automatically
- **Multi-Time Series Support**: Load and concatenate time series following xarray conventions
- **Multi-Level AMR**: Access any refinement level with automatic NaN filling for missing levels
- **Flexible Input Types**: Single files, file lists, or directories with pattern matching
- **Lazy Loading**: Efficient memory usage with dask-backed arrays
- **Automatic Sorting**: Time series automatically sorted by simulation time
- **Compatibility Validation**: Ensures consistent domains and fields across time steps

## Installation

```bash
# Install from source
git clone https://github.com/your-repo/xamrex.git
cd xamrex
pip install -e .
```

## Quick Start

### Single Plotfile

```python
import xarray as xr

# Load single AMReX plotfile
ds = xr.open_dataset('plt_00000', engine='amrex', level=0)

# Access data variables (lazy loaded)
temperature = ds['temp']
print(f"Shape: {temperature.shape}")
print(f"Time: {ds.attrs['current_time']}")
```

### Time Series (Multiple Files)

```python
import xarray as xr

# Method 1: Explicit file list
plotfiles = ['plt_00000', 'plt_01000', 'plt_02000']
ds = xr.open_dataset(plotfiles, engine='amrex', level=0)

# Method 2: Directory auto-discovery
ds = xr.open_dataset('simulation_output/', engine='amrex', level=0, pattern='plt_*')

# Method 3: Using utility functions
import xamrex
ds = xamrex.open_amrex_time_series(plotfiles, level=0)

print(f"Time series shape: {ds.dims}")
print(f"Time range: {ds.ocean_time.min().item()} to {ds.ocean_time.max().item()}")
```

### Multi-Level AMR

```python
# Load different refinement levels
ds_level0 = xr.open_dataset(plotfiles, engine='amrex', level=0)  # Base level
ds_level1 = xr.open_dataset(plotfiles, engine='amrex', level=1)  # Refined level

print(f"Level 0: {dict(ds_level0.sizes)}")  
print(f"Level 1: {dict(ds_level1.sizes)}")  # Higher resolution in x,y

# Missing levels automatically filled with NaN
# No errors if some time steps don't have the requested level
```

## Comprehensive API

### 1. Unified xarray Backend

The `amrex` engine automatically detects input type and handles appropriately:

```python
# Single plotfile directory
ds = xr.open_dataset("plt_00000", engine='amrex', level=0)

# List of plotfile directories (time series)
ds = xr.open_dataset(["plt_00000", "plt_01000"], engine='amrex', level=0)

# Directory containing plotfiles (auto-discovery)
ds = xr.open_dataset("simulation_data/", engine='amrex', level=0, pattern="plt_*")

# Custom patterns
ds = xr.open_dataset("data/", engine='amrex', level=1, pattern="sim_run_*")
```

### 2. Multi-Time Utility Functions

```python
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
```

### 3. Time Series Analysis

```python
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
```

### 4. Single-Level Utilities (Legacy Support)

```python
# Multi-level access utilities
levels = xamrex.open_amrex_levels("plt_00000", levels=[0, 1, 2])
summary = xamrex.create_level_summary("plt_00000")

# Level information
max_level = xamrex.get_max_level("plt_00000") 
available = xamrex.get_available_levels_from_file("plt_00000")

# Load specific levels
ds_level0 = xamrex.load_base_level("plt_00000")
ds_level1 = xamrex.load_level("plt_00000", level=1)
```

## Advanced Usage

### Custom Time Dimension

```python
# Use custom time dimension name
ds = xr.open_dataset(
    plotfiles, 
    engine='amrex', 
    level=0,
    time_dimension_name='time'  # Instead of default 'ocean_time'
)
```

### Custom Spatial Dimensions

```python
# Rename spatial coordinates
ds = xr.open_dataset(
    plotfiles, 
    engine='amrex', 
    level=0,
    dimension_names={'x': 'longitude', 'y': 'latitude', 'z': 'depth'}
)
```

### Memory Management

```python
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
```

## Data Structure

### Single Time Step

```python
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
```

### Time Series

```python
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
```

## Multi-Level AMR Support

### Automatic Level Detection

The backend automatically:
- Finds the first file with the requested level as a spatial template
- Uses that template for coordinate structure
- Fills missing levels with NaN values for time steps that don't have that level

```python
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
```

### Refinement Patterns

```python
# Level 0: Base resolution
ds_l0 = xr.open_dataset(files, engine='amrex', level=0)
print(f"Level 0: {dict(ds_l0.sizes)}")  # {'ocean_time': 3, 'z': 16, 'y': 15, 'x': 42}

# Level 1: Refined in x,y but not z  
ds_l1 = xr.open_dataset(files, engine='amrex', level=1) 
print(f"Level 1: {dict(ds_l1.sizes)}")  # {'ocean_time': 3, 'z': 16, 'y': 45, 'x': 126}
```

## Performance and Scalability

### Memory Efficiency
- **Lazy Loading**: Dask arrays mean large datasets don't overwhelm memory
- **Chunked Access**: Only load data when and where you need it
- **Efficient Concatenation**: Time series concatenation preserves lazy evaluation

### Large Time Series
```python
# Handle hundreds of time steps efficiently
all_files = xamrex.find_amrex_time_series("massive_simulation/", "plt_*")
print(f"Found {len(all_files)} files")  # Could be 1000+ files

# Still loads quickly (metadata only)
ds = xamrex.open_amrex_time_series(all_files, level=0)

# Extract just what you need
recent = ds.isel(ocean_time=slice(-10, None))  # Last 10 time steps
subset = recent.sel(x=slice(0.25, 0.75))      # Spatial subset
computed = subset.compute()                    # Only then load data
```

## Error Handling and Validation

```python
# Validate compatibility before loading
validation = xamrex.validate_time_series_compatibility(plotfiles, level=1)

if validation['compatible']:
    ds = xamrex.open_amrex_time_series(plotfiles, level=1)
else:
    print(f"Issues: {validation['issues']}")
    print(f"Available fields: {validation['fields']}")
```

## Migration from Version 0.4.x

The new API is fully backward compatible:

```python
# Old way (still works)
ds = xr.open_dataset("plt_00000", engine='amrex', level=0)

# New capabilities (no changes needed for single files)
ds = xr.open_dataset(["plt_00000", "plt_01000"], engine='amrex', level=0)  # Now works!
ds = xr.open_dataset("simulation_data/", engine='amrex', level=0)          # Now works!
```

## Requirements

- Python >= 3.8
- xarray >= 2023.1.0  
- numpy >= 1.20
- dask[array] >= 2021.1
- pandas >= 1.5.0

## Documentation

- [Multi-Time User Guide](README_MULTI_TIME.md) - Detailed multi-time functionality guide
- [Examples](examples/) - Example scripts and Jupyter notebooks
- [API Reference](docs/api.rst) - Complete API documentation

## Testing

```bash
# Run test suite
python -m pytest tests/

# Test multi-time functionality specifically  
python tests/test_multi_time.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

TBD

## Version History

- **v0.5.0**: Multi-time series support, unified backend, multi-level AMR
- **v0.4.x**: Single-file AMReX backend with lazy loading
