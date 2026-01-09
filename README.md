# xamrex: AMReX Backend for xarray

An xarray backend for reading AMReX plotfiles with native support for C-grid staggered variables, time series concatenation, and multi-level AMR data.

## Overview

`xamrex` registers the `amrex` backend to xarray for working with AMReX simulation data in Python. The package supports:
- **C-grid staggered variables** - Automatic detection and handling of rho, u, v, w, and psi grid points
- **2D and 3D variables** - Both vertically-integrated and full 3D fields
- **Multi-level AMR** - Proper coordinate scaling for refined levels
- **Time series** - Intelligent handling of multiple plotfiles with level masking
- **Lazy loading** - Dask-backed arrays for efficient memory usage

## Installation

```bash
# Install from source
git clone https://github.com/hetland/xamrex.git
cd xamrex
pip install -e .
```

## Quick Start

### Single Plotfile

```python
import xarray as xr

# Load single AMReX plotfile at Level 0
ds = xr.open_dataset('plt00000', engine='amrex', level=0)

# Access data variables (lazy loaded)
temperature = ds['temp']
salinity = ds['salt']
print(f"Variables: {list(ds.data_vars)}")
print(f"Dimensions: {dict(ds.dims)}")
```

### Time Series (Multiple Files)

```python
import xarray as xr
from glob import glob

# Load multiple plotfiles as time series
files = sorted(glob('simulation/plt*'))
ds = xr.open_dataset(files, engine='amrex', level=0)

# Time dimension automatically added
print(f"Time steps: {len(ds.ocean_time)}")
print(f"Time range: {ds.ocean_time.values[0]} to {ds.ocean_time.values[-1]}")
```

### Multi-Level AMR

```python
# Load different refinement levels
ds_level0 = xr.open_dataset(files, engine='amrex', level=0)  # Base level
ds_level1 = xr.open_dataset(files, engine='amrex', level=1)  # Refined level

print(f"Level 0: {dict(ds_level0.dims)}")  
print(f"Level 1: {dict(ds_level1.dims)}")  # Higher resolution

# Coordinates are properly scaled for refinement
print(f"Level 0 cell size: {(ds_level0.x_rho[1] - ds_level0.x_rho[0]).values:.1f}m")
print(f"Level 1 cell size: {(ds_level1.x_rho[1] - ds_level1.x_rho[0]).values:.1f}m")
```

## C-Grid Support

The backend automatically detects and handles staggered C-grid variables:

```python
ds = xr.open_dataset('plt00000', engine='amrex', level=0)

# Rho-points (cell centers)
temp = ds['temp']     # dimensions: (ocean_time, z_rho, y_rho, x_rho)
salt = ds['salt']

# U-points (x-faces)
u_vel = ds['u_vel']   # dimensions: (ocean_time, z_rho, y_u, x_u)

# V-points (y-faces)
v_vel = ds['v_vel']   # dimensions: (ocean_time, z_rho, y_v, x_v)

# W-points (z-faces)
w_vel = ds['w_vel']   # dimensions: (ocean_time, z_w, y_w, x_w)

# 2D variables (barotropic)
zeta = ds['zeta']     # dimensions: (ocean_time, y_rho, x_rho)
```

### xgcm Integration

The backend includes xgcm-compatible metadata for grid-aware operations:

```python
import xgcm

# Create xgcm Grid object
grid = xgcm.Grid(ds, periodic=False)

# Perform grid-aware operations
temp_at_u = grid.interp(ds.temp, 'X')  # Interpolate to u-points
div_h = grid.diff(ds.u_vel, 'X') + grid.diff(ds.v_vel, 'Y')  # Horizontal divergence
```

## AMR Level Handling

### Automatic Level Detection

The backend automatically handles AMR levels with proper coordinate scaling:

```python
# Level 0: Base resolution (e.g., 2000m grid spacing)
ds0 = xr.open_dataset(files, engine='amrex', level=0)

# Level 1: 3x refinement (e.g., 667m grid spacing)  
ds1 = xr.open_dataset(files, engine='amrex', level=1)

# Coordinates reflect actual physical locations
# Level 1 covers a subregion with finer resolution
```

### Level Masking

When a level doesn't exist at certain time steps, those times are filled with NaN:

```python
# Example: Mixed-level time series
# plt00000: only level 0 exists
# plt00100: levels 0 and 1 exist
# plt00200: levels 0 and 1 exist

ds_level1 = xr.open_dataset(['plt00000', 'plt00100', 'plt00200'], 
                           engine='amrex', level=1)

# Result: Level 1 dataset with:
# - Time step 0: All NaN (level 1 doesn't exist)
# - Time steps 1-2: Valid data where level 1 exists
```

## Features

### Lazy Loading with Dask
- Large datasets are handled efficiently with lazy evaluation
- Only load data when needed for computation
- Supports out-of-core computation for datasets larger than memory

### 2D and 3D Variables
- Automatically detects dimensionality from AMReX headers
- 2D variables: `(ocean_time, y, x)`
- 3D variables: `(ocean_time, z, y, x)`

### Coordinate Generation
- Physical coordinates generated from domain bounds
- Proper staggering for C-grid points
- Refinement-aware scaling for AMR levels

## Advanced Usage

### Drop Variables to Save Memory

```python
# Load only specific variables
ds = xr.open_dataset(
    files, 
    engine='amrex', 
    level=0,
    drop_variables=['salt', 'w_vel']
)
```

### Working with Large Time Series

```python
# Load metadata only (fast)
ds = xr.open_dataset(files, engine='amrex', level=0)

# Select subset before computing
subset = ds.isel(ocean_time=slice(0, 10))  # First 10 time steps
subset = subset.sel(z_rho=0)               # Surface only
result = subset.compute()                   # Load only this subset
```

## Performance

- **Memory Efficient**: Dask arrays enable lazy loading
- **Scalable**: Handle hundreds of time steps efficiently  
- **Fast Metadata**: Quick to open without loading data
- **Chunked Access**: Load only what you need

## Requirements

- Python >= 3.8
- xarray >= 2023.1.0  
- numpy >= 1.20
- dask >= 2021.1
- pandas >= 1.5.0

## Documentation

Full documentation available at: [https://xamrex.readthedocs.io](https://xamrex.readthedocs.io)

## Testing

```bash
# Run test suite
python -m pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE.txt for details

## Citation

If you use this package in your research, please cite:

```bibtex
@software{xamrex,
  author = {Hetland, Robert},
  title = {xamrex: AMReX Backend for xarray},
  year = {2025},
  url = {https://github.com/hetland/xamrex}
}
