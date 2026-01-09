"""
Xarray backend for C-grid AMReX plotfiles with automatic grid detection.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable, List, Dict
import os

import numpy as np
import xarray as xr
from xarray.backends.common import (
    AbstractDataStore,
    BackendArray,
    BackendEntrypoint,
)
from xarray.core.variable import Variable

from .metadata import AMReXBasicMeta, AMReXMultiGridMeta
from .coordinates import CGridCoordinateGenerator
from .fab_loader import FABMetadata, FABLoader, MaskedFABLoader


class AMReXCGridStore(AbstractDataStore):
    """
    Xarray data store for C-grid AMReX plotfiles.
    
    Supports:
    - Multiple grid types (rho, u, v, w, psi)
    - 2D and 3D variables
    - Multi-file time series
    - Level masking across time
    - xgcm-compatible metadata
    """
    
    def __init__(self, plotfile_paths: List[Path], level: int = 0):
        """
        Initialize C-grid store.
        
        Parameters
        ----------
        plotfile_paths : list of Path
            Plotfile directories (single or multiple for time series)
        level : int, default 0
            AMR level to load
        """
        self.plotfile_paths = [Path(p) for p in plotfile_paths]
        self.level = level
        self.is_time_series = len(self.plotfile_paths) > 1
        
        # Parse metadata
        if self.is_time_series:
            self.meta = AMReXMultiGridMeta(self.plotfile_paths)
        else:
            # Single file - create simple metadata
            self.basic_meta = AMReXBasicMeta(self.plotfile_paths[0])
            # Wrap in multi-grid structure for consistency
            self.meta = AMReXMultiGridMeta(self.plotfile_paths)
        
        # Initialize coordinate generator with level and refinement info
        self.coord_gen = CGridCoordinateGenerator(
            self.meta.basic_meta.domain_left_edge,
            self.meta.basic_meta.domain_right_edge,
            self.meta.basic_meta.level_dimensions,
            self.meta.basic_meta.dimensionality,
            refinement_factors=self.meta.basic_meta.ref_factors,
            level=level
        )
        
        # Check if requested level ever exists
        if level not in self.meta.level_availability:
            raise ValueError(
                f"Level {level} never appears in provided plotfiles. "
                f"Available levels: {list(self.meta.level_availability.keys())}"
            )
    
    def get_variables(self) -> Dict[str, Variable]:
        """Return all variables including coordinates and data variables."""
        variables = {}
        
        # Create time coordinate
        variables['ocean_time'] = Variable(
            dims=('ocean_time',),
            data=self.meta.time_values,
            attrs={
                'long_name': 'time',
                'units': 'seconds',  # TODO: extract from AMReX if available
            }
        )
        
        # Generate z-coordinates once from level dimensions (shared across all grids)
        if self.meta.basic_meta.dimensionality == 3:
            # Generate z_rho coordinate from level dimensions
            level_dims = self.meta.basic_meta.level_dimensions[self.level]
            nz_rho = level_dims[2] if len(level_dims) > 2 else 32
            
            z_rho_coord = self.coord_gen._generate_z_coordinate(nz_rho, 'rho')
            variables['z_rho'] = Variable(
                dims=('z_rho',),
                data=z_rho_coord,
                attrs=self.coord_gen._get_coordinate_attrs('z_rho', 'Z', 'rho')
            )
            
            # Generate z_w coordinate from WFace dimensions if it exists
            if 'WFace' in self.meta.all_grids:
                wface_dims = self.meta.all_grids['WFace']['dimensions']
                nz_w = wface_dims[2] if len(wface_dims) > 2 else nz_rho + 1
                z_w_coord = self.coord_gen._generate_z_coordinate(nz_w, 'w')
                variables['z_w'] = Variable(
                    dims=('z_w',),
                    data=z_w_coord,
                    attrs=self.coord_gen._get_coordinate_attrs('z_w', 'Z', 'w')
                )
        
        # Generate horizontal coordinates for each discovered grid type
        # Use level-specific dimensions, not cached dimensions from all_grids
        from .grid_detector import GridDetector
        detector = GridDetector()
        level_grids = detector.detect_grids(self.plotfile_paths[0], self.level)
        
        coords_created = set()
        for dir_name, grid_info in level_grids.items():
            grid_type = grid_info['grid_type']
            
            # Generate only x and y coordinates (z is shared)
            coords = self.coord_gen.generate_xy_coordinates(
                grid_type, grid_dimensions=grid_info['dimensions']
            )
            
            # Add coordinates (avoid duplicates)
            for coord_name, (dim_name, coord_array, attrs) in coords.items():
                if coord_name not in coords_created:
                    variables[coord_name] = Variable(
                        dims=(dim_name,),
                        data=coord_array,
                        attrs=attrs
                    )
                    coords_created.add(coord_name)
        
        # Get reference z-dimensions for coordinate assignment
        level_dims = self.meta.basic_meta.level_dimensions[self.level]
        nz_rho = level_dims[2] if len(level_dims) > 2 else 32
        nz_w = nz_rho + 1  # w-points have one extra vertical level
        
        # Create data variables
        for var_name in self.meta.basic_meta.field_list:
            # Determine which grid this variable belongs to
            dir_name = self.meta.get_variable_grid(var_name)
            grid_info = self.meta.get_grid_info(dir_name)
            
            if not grid_info:
                print(f"Warning: No grid info for variable {var_name}, skipping")
                continue
            
            grid_type = grid_info['grid_type']
            is_2d = grid_info['dimensionality'] == 2
            
            # Create lazy array
            lazy_array = self._create_lazy_array(var_name, dir_name, grid_info)
            
            # Get dimension names with automatic z-coordinate detection
            if is_2d:
                dim_names = self.coord_gen.get_dimension_names(grid_type, is_2d)
            else:
                # For 3D variables, determine correct z-coordinate based on actual grid dimensions
                grid_dims = grid_info['dimensions']
                nz_actual = grid_dims[2] if len(grid_dims) > 2 else nz_rho
                
                # Get base dimension names (will have z_rho by default)
                dim_names = list(self.coord_gen.get_dimension_names(grid_type, is_2d))
                
                # Replace z-coordinate based on actual vertical dimension
                if nz_actual == nz_w:
                    # This grid has w-points in vertical - use z_w
                    dim_names[1] = 'z_w'
                # else: keep z_rho (default)
            
            # Create variable
            variables[var_name] = Variable(
                dims=dim_names,
                data=lazy_array,
                attrs={
                    'long_name': var_name,
                    'grid': grid_type,
                    'directory': dir_name,
                    '_FillValue': np.nan,
                }
            )
        
        return variables
    
    def _create_lazy_array(self, var_name: str, dir_name: str, 
                          grid_info: Dict) -> Any:
        """
        Create lazy dask array for a variable.
        
        Handles both single file and time series cases, with masking.
        """
        import dask.array as da
        
        # Get the component index for this variable within its grid
        var_index = self.meta.basic_meta.variable_to_component_index.get(var_name, 0)
        
        # Get dimensions for the REQUESTED level, not from cached grid_info
        # grid_info['dimensions'] may be from a different level!
        # Instead, we need to detect the grid at the current level
        is_2d = grid_info['dimensionality'] == 2
        
        # Get dimensions by actually reading the grid at this level
        # We'll use the first plotfile to determine dimensions
        from .grid_detector import GridDetector
        detector = GridDetector()
        level_grids = detector.detect_grids(self.plotfile_paths[0], self.level)
        
        if dir_name in level_grids:
            dimensions = level_grids[dir_name]['dimensions']
        else:
            # Fallback to grid_info dimensions (shouldn't happen)
            dimensions = grid_info['dimensions']
        
        # Use dimensions as-is from grid detector
        # No stagger adjustments needed - they're already included!
        if is_2d:
            # 2D: dimensions are (nx, ny)
            spatial_shape = (dimensions[1], dimensions[0])  # (ny, nx) in C-order
        else:
            # 3D: dimensions are (nx, ny, nz)
            spatial_shape = (dimensions[2], dimensions[1], dimensions[0])  # (nz, ny, nx)
        
        if self.is_time_series:
            # Time series: concatenate along time
            time_arrays = []
            
            for time_idx, pf_path in enumerate(self.plotfile_paths):
                # Check if level exists at this timestep
                if self.meta.is_level_available(self.level, time_idx):
                    # Load data
                    full_shape = (1,) + spatial_shape
                    try:
                        fab_meta = FABMetadata(
                            pf_path, self.level, dir_name,
                            grid_info['num_components'],
                            grid_info['dimensionality']
                        )
                        loader = FABLoader(
                            pf_path, self.level, dir_name,
                            var_index, fab_meta, full_shape
                        )
                        time_arrays.append(loader.create_dask_array())
                    except Exception as e:
                        print(f"Warning: Failed to load {var_name} at time {time_idx}: {e}")
                        # Create masked array
                        masked_loader = MaskedFABLoader(full_shape)
                        time_arrays.append(masked_loader.create_dask_array())
                else:
                    # Level doesn't exist - create masked array
                    full_shape = (1,) + spatial_shape
                    masked_loader = MaskedFABLoader(full_shape)
                    time_arrays.append(masked_loader.create_dask_array())
            
            # Concatenate along time dimension
            return da.concatenate(time_arrays, axis=0)
        else:
            # Single file
            full_shape = (1,) + spatial_shape
            try:
                fab_meta = FABMetadata(
                    self.plotfile_paths[0], self.level, dir_name,
                    grid_info['num_components'],
                    grid_info['dimensionality']
                )
                loader = FABLoader(
                    self.plotfile_paths[0], self.level, dir_name,
                    var_index, fab_meta, full_shape
                )
                return loader.create_dask_array()
            except Exception as e:
                print(f"Warning: Failed to load {var_name}: {e}")
                masked_loader = MaskedFABLoader(full_shape)
                return masked_loader.create_dask_array()
    
    def get_attrs(self) -> Dict:
        """Return global attributes including xgcm grid metadata."""
        attrs = {
            'title': f'C-grid AMReX Plotfile Level {self.level}',
            'level': self.level,
            'max_level_ever': self.meta.max_level_ever,
            'dimensionality': self.meta.basic_meta.dimensionality,
            'domain_left_edge': self.meta.basic_meta.domain_left_edge.tolist(),
            'domain_right_edge': self.meta.basic_meta.domain_right_edge.tolist(),
        }
        
        if self.is_time_series:
            attrs['time_steps'] = len(self.plotfile_paths)
            attrs['time_range'] = [float(self.meta.time_values[0]), 
                                  float(self.meta.time_values[-1])]
            attrs['source_files'] = [p.name for p in self.plotfile_paths]
        else:
            attrs['plotfile'] = str(self.plotfile_paths[0])
            attrs['time'] = float(self.meta.basic_meta.current_time)
        
        # Add xgcm grid topology
        attrs['xgcm-Grid'] = self._create_xgcm_grid_spec()
        
        return attrs
    
    def _create_xgcm_grid_spec(self) -> Dict:
        """Create xgcm grid specification."""
        grid_spec = {}
        
        # X-axis
        if any('u' == g['grid_type'] for g in self.meta.all_grids.values()):
            grid_spec['X'] = {
                'center': 'x_rho',
                'left': 'x_u',
            }
            if any('psi' == g['grid_type'] for g in self.meta.all_grids.values()):
                grid_spec['X']['outer'] = 'x_psi'
        
        # Y-axis
        if any('v' == g['grid_type'] for g in self.meta.all_grids.values()):
            grid_spec['Y'] = {
                'center': 'y_rho',
                'left': 'y_v',
            }
            if any('psi' == g['grid_type'] for g in self.meta.all_grids.values()):
                grid_spec['Y']['outer'] = 'y_psi'
        
        # Z-axis (if 3D)
        if self.meta.basic_meta.dimensionality == 3:
            if any('w' == g['grid_type'] for g in self.meta.all_grids.values()):
                grid_spec['Z'] = {
                    'center': 'z_rho',
                    'outer': 'z_w',
                }
        
        return grid_spec


class AMReXCGridEntrypoint(BackendEntrypoint):
    """
    Xarray backend entrypoint for C-grid AMReX plotfiles.
    """
    
    def open_dataset(
        self,
        filename_or_obj: str | os.PathLike | List,
        *,
        drop_variables: str | Iterable[str] | None = None,
        level: int = 0,
        **kwargs
    ) -> xr.Dataset:
        """
        Open AMReX plotfile(s) as xarray Dataset.
        
        Parameters
        ----------
        filename_or_obj : str, Path, or list
            Single plotfile directory or list of plotfile directories
        drop_variables : str or iterable, optional
            Variables to exclude
        level : int, default 0
            AMR level to load
        **kwargs
            Additional arguments (for compatibility)
            
        Returns
        -------
        xr.Dataset
            Dataset with C-grid variables
        """
        # Normalize input to list
        if isinstance(filename_or_obj, (list, tuple)):
            plotfile_paths = [Path(p) for p in filename_or_obj]
        else:
            plotfile_paths = [Path(filename_or_obj)]
        
        # Create store
        store = AMReXCGridStore(plotfile_paths, level=level)
        
        # Get variables and attributes
        variables = store.get_variables()
        attributes = store.get_attrs()
        
        # Filter dropped variables
        if drop_variables:
            if isinstance(drop_variables, str):
                drop_variables = [drop_variables]
            for var in drop_variables:
                variables.pop(var, None)
        
        # Separate coordinates from data variables
        coord_names = {'ocean_time', 'x_rho', 'y_rho', 'z_rho', 'z_w',
                      'x_u', 'y_u', 'x_v', 'y_v', 'x_psi', 'y_psi'}
        
        coords = {}
        data_vars = {}
        
        for name, var in variables.items():
            if name in coord_names:
                coords[name] = var
            else:
                data_vars[name] = var
        
        # Create dataset
        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=attributes
        )
    
    def guess_can_open(self, filename_or_obj) -> bool:
        """Check if this looks like an AMReX plotfile."""
        try:
            if isinstance(filename_or_obj, (list, tuple)):
                path = Path(filename_or_obj[0])
            else:
                path = Path(filename_or_obj)
            
            if not path.is_dir():
                return False
            
            # Check for AMReX structure
            header = path / 'Header'
            level_0 = path / 'Level_0'
            
            return header.exists() and level_0.is_dir()
        except Exception:
            return False
    
    open_dataset_parameters = ["filename_or_obj", "drop_variables", "level"]
    description = "C-grid AMReX plotfile backend with automatic grid detection"
    url = "https://github.com/hetland/xamrex"
