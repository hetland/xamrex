## xamrex2 - C-Grid AMReX Plotfile Reader

Enhanced xarray backend for reading AMReX plotfiles with staggered C-grid support.

### Features

- **Automatic grid detection**: Discovers all grid types (rho, u, v, w, psi points)
- **C-grid coordinates**: Proper naming (`x_rho`, `x_u`, `x_v`, etc.)
- **2D and 3D variables**: Handles both depth-averaged and full 3D fields
- **Multi-file time series**: Concatenates multiple plotfiles along time dimension
- **Level masking**: Returns NaN when requested level doesn't exist at a timestep
- **Lazy loading**: Uses dask for memory-efficient data access
- **xgcm compatible**: Includes grid metadata for automatic regridding

### Installation

```python
# Add to your Python path or install in development mode
pip install -e .
```

### Usage

#### Single file

```python
import xarray as xr

ds = xr.open_dataset('ocean_out/plt00000', engine='xamrex2', level=0)
print(ds)
```

#### Time series

```python
import xarray as xr
from pathlib import Path

plotfiles = sorted(Path('ocean_out').glob('plt*'))
ds = xr.open_dataset(plotfiles, engine='xamrex2', level=0)
print(ds)
```

#### With xgcm

```python
import xarray as xr
import xgcm

ds = xr.open_dataset('ocean_out/plt00000', engine='xamrex2', level=0)
grid = xgcm.Grid(ds, periodic=False)

# Interpolate u-velocity to rho-points
u_rho = grid.interp(ds.u_vel, 'X')
```

### Architecture

- `grid_detector.py`: Automatic detection of grid types from directory structure
- `metadata.py`: Multi-grid metadata parsing with level availability tracking
- `coordinates.py`: C-grid coordinate generation with proper staggering
- `fab_loader.py`: FAB data loading with lazy evaluation and masking
- `backend.py`: Xarray backend integration

### Coordinate System

The package uses ROMS-style C-grid naming:

- **rho-points** (`x_rho`, `y_rho`, `z_rho`): Cell centers
- **u-points** (`x_u`, `y_u`): X-face centers (staggered in x)
- **v-points** (`x_v`, `y_v`): Y-face centers (staggered in y)
- **w-points** (`z_w`): Z-face centers (staggered in z)
- **psi-points** (`x_psi`, `y_psi`): Corner points

Each coordinate includes xgcm-compatible metadata (`c_grid_axis_shift`) for automatic interpolation.

### Testing

Run unit tests:
```bash
pytest tests/test_xamrex2.py -v
```

Run integration tests:
```bash
python test_integration.py
