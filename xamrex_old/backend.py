"""
Single-level AMReX backend with lazy dask loading.
Each dataset represents a single AMR level with full domain dimensions and masked arrays.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Hashable, Iterable
import os
import struct

import numpy as np
import dask.array as da
import xarray as xr
from xarray.backends.common import (
    AbstractDataStore,
    BackendArray,
    BackendEntrypoint,
    ReadBuffer,
)
from xarray.core.variable import Variable

from .AMReX_array import AMReXDatasetMeta, AMReXFabsMetaSingleLevel
from .refinement import create_refinement_handler


class AMReXLazyArray(BackendArray):
    """
    Simplified lazy array wrapper for AMReX data using dask for memory efficiency.
    Represents a single field at a single AMR level.
    """
    
    def __init__(self, fplt: Path, level: int, field_name: str, 
                 meta: AMReXDatasetMeta, fab_meta: AMReXFabsMetaSingleLevel):
        self.fplt = fplt
        self.level = level
        self.field_name = field_name
        self.meta = meta
        self.fab_meta = fab_meta
        self.field_idx = meta.field_list.index(field_name)
        
        # Use the new RefinementHandler for all refinement calculations
        self.refinement_handler = create_refinement_handler(meta)
        
        # Get array shape using the refinement handler
        self.shape = self.refinement_handler.get_full_shape(level, include_time=True)
        self.dtype = np.float64
        
        # Create a dask array that will actually load data when accessed
        self.dask_array = self._create_dask_array()
    
    @property
    def itemsize(self):
        """Return the itemsize of the dtype."""
        return self.dtype.itemsize
    
    @property
    def size(self):
        """Return the total number of elements."""
        return int(np.prod(self.shape))
    
    @property
    def nbytes(self):
        """Return the total number of bytes."""
        return self.size * self.itemsize
    
    def _create_dask_array(self):
        """Create a dask array that loads AMReX data on demand."""
        # Create a dask array using from_delayed with the actual data loading function
        from dask import delayed
        
        @delayed
        def load_amrex_data():
            return self._load_full_field_data()
        
        # Create dask array from the delayed computation
        lazy_data = load_amrex_data()
        return da.from_delayed(lazy_data, shape=self.shape, dtype=self.dtype)
    
    def _load_full_field_data(self):
        """Load the complete field data from all FABs and assemble into full grid."""
        # FIXED ISSUE 2: Initialize full grid with NaN for missing values instead of zeros
        full_data = np.full(self.shape, np.nan, dtype=np.float64)
        
        # Load data from each FAB and place it in the correct location
        for fab_idx, fab_row in self.fab_meta.metadata.iterrows():
            # Read data from this FAB
            fab_data = self._read_fab_data(fab_idx)
            
            # Get index ranges for this FAB
            lo_i, lo_j, lo_k = fab_row['lo_i'], fab_row['lo_j'], fab_row['lo_k']
            hi_i, hi_j, hi_k = fab_row['hi_i'], fab_row['hi_j'], fab_row['hi_k']
            
            # Direct mapping using actual FAB indices
            if self.meta.dimensionality == 3:
                # 3D case: (time, z, y, x)
                full_data[0, lo_k:hi_k, lo_j:hi_j, lo_i:hi_i] = fab_data
            else:
                # 2D case: (time, y, x)
                full_data[0, lo_j:hi_j, lo_i:hi_i] = fab_data
        
        return full_data
    
    def _read_fab_data(self, fab_idx):
        """Read data for a specific field from a specific FAB."""
        fab_row = self.fab_meta.metadata.loc[fab_idx]
        fab_file = self.fplt / f"Level_{self.level}" / fab_row['filename']
        
        # Calculate fab dimensions
        di, dj, dk = fab_row['di'], fab_row['dj'], fab_row['dk']
        cells_per_fab = di * dj * dk
        
        # Open fab file and seek to the data for this field
        with open(fab_file, 'rb') as f:
            # Skip to the byte offset for this fab
            f.seek(fab_row['byte_offset'])
            
            # Skip the metadata line to get to data start
            f.readline()
            data_start = f.tell()
            
            # FIXED: AMReX stores data in blocked format: all field0 data, then all field1 data, etc.
            # Calculate the start position for this field's data block
            field_start = data_start + self.field_idx * cells_per_fab * 8  # 8 bytes per float64
            
            # Seek to the start of this field's data block
            f.seek(field_start)
            
            # Read all data for this field at once
            field_bytes = f.read(cells_per_fab * 8)
            
            if len(field_bytes) == cells_per_fab * 8:
                # Unpack all values at once (much more efficient)
                field_data = struct.unpack(f'<{cells_per_fab}d', field_bytes)
            else:
                # Fallback: read individual values
                field_data = []
                f.seek(field_start)
                for cell_idx in range(cells_per_fab):
                    value_bytes = f.read(8)
                    if len(value_bytes) == 8:
                        value = struct.unpack('<d', value_bytes)[0]
                        field_data.append(value)
                    else:
                        field_data.append(0.0)
            
            # Reshape to fab dimensions (in C order: k,j,i since AMReX uses C ordering in files)
            if self.meta.dimensionality == 3:
                fab_data = np.array(field_data, dtype=np.float64).reshape((dk, dj, di))
            else:
                fab_data = np.array(field_data, dtype=np.float64).reshape((dj, di))
                
            return fab_data
    
    def get_array(self):
        """Return the underlying dask array."""
        return self.dask_array
    
    def __getitem__(self, key):
        """Get array slice with proper xarray indexing support."""
        # Handle xarray indexing objects
        if hasattr(key, 'tuple'):
            # BasicIndexer and other xarray indexing objects
            key = key.tuple
        elif hasattr(key, '__getitem__') and not isinstance(key, (tuple, slice, int, np.integer)):
            # Convert other indexing objects to tuple
            try:
                key = tuple(key)
            except (TypeError, ValueError):
                # If conversion fails, try to extract the indexing information
                if hasattr(key, 'indices'):
                    key = key.indices
                else:
                    # Fallback to identity slicing
                    key = tuple(slice(None) for _ in self.shape)
        
        return self.dask_array[key]


class AMReXSingleLevelStore(AbstractDataStore):
    """
    Data store for a single AMR level from an AMReX plotfile.
    """
    
    def __init__(self, plotfile_path: Path, level: int = 0, time_dimension_name: str = None, dimension_names: dict = None):
        self.plotfile_path = Path(plotfile_path)
        self.level = level
        
        # Parse metadata with configurable dimension names
        self.meta = AMReXDatasetMeta(plotfile_path, time_dimension_name, dimension_names)
        
        # Validate level exists
        if level > self.meta.max_level:
            raise ValueError(f"Level {level} not available. Max level: {self.meta.max_level}")
        
        # Get fab metadata for this level
        try:
            self.fab_meta = AMReXFabsMetaSingleLevel(
                plotfile_path, self.meta.n_fields, self.meta.dimensionality, level
            )
        except FileNotFoundError:
            raise ValueError(f"Level {level} directory not found in {plotfile_path}")
    
    def get_variables(self):
        """Return variables for all fields at this level."""
        variables = {}
        
        # Use RefinementHandler for coordinate calculations
        refinement_handler = create_refinement_handler(self.meta)
        
        # Get time dimension name
        time_dim_name = getattr(self.meta, 'time_dimension_name', 'ocean_time')
        
        # Create dimension names with the configurable time dimension
        if self.meta.dimensionality == 3:
            dim_names = [time_dim_name, 'z', 'y', 'x']  # Add time dimension
        else:
            dim_names = [time_dim_name, 'y', 'x']       # Add time dimension for 2D
        
        # Create time coordinate (singleton dimension)
        variables[time_dim_name] = Variable(
            dims=(time_dim_name,),
            data=np.array([self.meta.current_time]),
            attrs={
                'long_name': 'Time',
                'units': 'unknown',  # TODO: extract units from AMReX if available
            }
        )
        
        # Generate spatial coordinates using RefinementHandler
        coord_arrays = refinement_handler.get_coordinate_arrays(
            self.level, self.meta.domain_left_edge, self.meta.domain_right_edge
        )
        
        # Create spatial coordinate variables
        refinement_factors = refinement_handler.get_refinement_factors(self.level)
        for dim_name, coord_array in coord_arrays.items():
            dim_idx = ['x', 'y', 'z'].index(dim_name)
            refinement_factor = refinement_factors[dim_idx]
            
            # Calculate spacing info
            if len(coord_array) > 1:
                spacing = coord_array[1] - coord_array[0]
            else:
                spacing = 0.0
            
            base_spacing = spacing * refinement_factor
            
            variables[dim_name] = Variable(
                dims=(dim_name,),
                data=coord_array,
                attrs={
                    'long_name': f'{dim_name.upper()} coordinate',
                    'units': 'unknown',  # TODO: extract from AMReX if available
                    'spacing': float(spacing),
                    'base_spacing': float(base_spacing),
                    'refinement_factor': int(refinement_factor),
                }
            )
        
        # Create data variables for each field
        base_refinement = refinement_handler.base_refinement
        for field in self.meta.field_list:
            # Create lazy array wrapper
            lazy_array = AMReXLazyArray(
                self.plotfile_path, self.level, field, self.meta, self.fab_meta
            )
            
            # Use the dask array directly instead of the wrapper
            variables[field] = Variable(
                dims=dim_names,
                data=lazy_array.dask_array,
                attrs={
                    'long_name': field,
                    'level': int(self.level),
                    'refinement_factor': int(base_refinement ** self.level),
                    '_FillValue': np.nan,
                }
            )
        
        return variables
    
    def get_dimensions(self):
        """Return dimensions for this level using header data."""
        # Use actual dimensions from header instead of calculations
        level_dims = self.meta.get_level_dimensions(self.level)
        
        # Use Fortran order dimension names to match data layout
        if self.meta.dimensionality == 3:
            dim_names = ['z', 'y', 'x']  # Fortran order for 3D
            coord_indices = [2, 1, 0]    # Map to original x,y,z indices
        else:
            dim_names = ['y', 'x']       # Fortran order for 2D  
            coord_indices = [1, 0]       # Map to original x,y indices
            
        return {dim: level_dims[coord_indices[i]] for i, dim in enumerate(dim_names)}
    
    def get_attrs(self):
        """Return global attributes."""
        return {
            'title': f'AMReX Plotfile Level {self.level}: {self.plotfile_path.name}',
            'plotfile_path': str(self.plotfile_path),
            'level': int(self.level),
            'max_level': int(self.meta.max_level),
            'current_time': float(self.meta.current_time),
            'dimensionality': int(self.meta.dimensionality),
            'geometry': str(self.meta.geometry),
            'domain_left_edge': self.meta.domain_left_edge.tolist(),
            'domain_right_edge': self.meta.domain_right_edge.tolist(),
            'refinement_factor': int(self.meta.ref_factors[0] ** self.level),
            'base_refinement_factors': list(self.meta.ref_factors),
            'fields': self.meta.field_list,
        }


class AMReXMultiTimeStore(AbstractDataStore):
    """
    Data store for multiple AMReX plotfiles concatenated along time dimension.
    """
    
    def __init__(self, plotfile_paths: list, level: int = 0, 
                 time_dimension_name: str = None, dimension_names: dict = None):
        """
        Initialize multi-time store.
        
        Parameters
        ----------
        plotfile_paths : list of str or Path
            List of paths to AMReX plotfile directories
        level : int, default 0
            AMR level to load from each plotfile
        time_dimension_name : str, optional
            Name for the time dimension (default: 'ocean_time')
        dimension_names : dict, optional
            Custom dimension names
        """
        self.plotfile_paths = [Path(p) for p in plotfile_paths]
        self.level = level
        self.time_dimension_name = time_dimension_name or 'ocean_time'
        self.dimension_names = dimension_names or {}
        
        # Validate all plotfiles exist and can be opened (but don't require all to have the level)
        self.stores = []
        self.times = []
        
        for path in self.plotfile_paths:
            try:
                # Just create a basic store to get metadata and time, not necessarily for this level
                meta_store = AMReXSingleLevelStore(path, 0, time_dimension_name, dimension_names)  # Always use level 0 for metadata
                self.stores.append(meta_store)
                self.times.append(meta_store.meta.current_time)
            except Exception as e:
                raise ValueError(f"Failed to open plotfile {path}: {e}")
        
        # Sort by time to ensure proper ordering
        sorted_indices = np.argsort(self.times)
        self.stores = [self.stores[i] for i in sorted_indices]
        self.times = [self.times[i] for i in sorted_indices]
        self.plotfile_paths = [self.plotfile_paths[i] for i in sorted_indices]
        
        # Use first store as reference for metadata
        self.reference_store = self.stores[0]
        
        # Validate compatibility across all stores
        self._validate_compatibility()
    
    def _validate_compatibility(self):
        """Validate that all plotfiles are compatible for concatenation."""
        ref_meta = self.reference_store.meta
        
        # Find the first file that has the requested level to use as spatial template
        self.level_template_store = None
        for store in self.stores:
            if store.meta.max_level >= self.level:
                try:
                    # Try to create the store for this level to ensure it works
                    test_store = AMReXSingleLevelStore(
                        store.plotfile_path, self.level, 
                        self.time_dimension_name, self.dimension_names
                    )
                    self.level_template_store = test_store
                    break
                except Exception:
                    continue
        
        if self.level_template_store is None:
            raise ValueError(f"Level {self.level} not available in any of the provided files")
        
        for i, store in enumerate(self.stores[1:], 1):
            meta = store.meta
            
            # Check spatial dimensions
            if meta.dimensionality != ref_meta.dimensionality:
                raise ValueError(f"Dimensionality mismatch: {meta.dimensionality} vs {ref_meta.dimensionality}")
            
            # Check field compatibility (only for files that have data)
            if set(meta.field_list) != set(ref_meta.field_list):
                raise ValueError(f"Field list mismatch in file {i}")
            
            # Check domain compatibility
            if not np.allclose(meta.domain_left_edge, ref_meta.domain_left_edge):
                raise ValueError(f"Domain left edge mismatch in file {i}")
            
            if not np.allclose(meta.domain_right_edge, ref_meta.domain_right_edge):
                raise ValueError(f"Domain right edge mismatch in file {i}")
            
            # Note: We no longer require all files to have the requested level
            # Missing levels will be filled with NaN values
    
    def get_variables(self):
        """Return variables concatenated along time dimension."""
        variables = {}
        
        # Get variables from level template store to establish spatial structure
        template_vars = self.level_template_store.get_variables()
        
        # Create time coordinate from all files
        time_values = np.array(self.times)
        variables[self.time_dimension_name] = Variable(
            dims=(self.time_dimension_name,),
            data=time_values,
            attrs={
                'long_name': 'Time',
                'units': 'unknown',  # TODO: extract units from AMReX if available
            }
        )
        
        # Add spatial coordinates from level template (same for all time steps at this level)
        for name, var in template_vars.items():
            if name != self.time_dimension_name and name in ['x', 'y', 'z']:
                variables[name] = var
        
        # Concatenate data variables along time dimension
        for field_name in self.level_template_store.meta.field_list:
            # Collect data arrays from all time steps
            data_arrays = []
            
            for store in self.stores:
                # Check if this store has the requested level
                if store.meta.max_level >= self.level:
                    try:
                        # Try to load this level
                        level_store = AMReXSingleLevelStore(
                            store.plotfile_path, self.level,
                            self.time_dimension_name, self.dimension_names
                        )
                        store_vars = level_store.get_variables()
                        
                        if field_name in store_vars:
                            # Convert Variable to DataArray for concatenation
                            var = store_vars[field_name]
                            data_array = xr.DataArray(var.data, dims=var.dims, attrs=var.attrs)
                            data_arrays.append(data_array)
                        else:
                            # Field missing in this file - create NaN DataArray
                            template_var = template_vars[field_name]
                            nan_data = np.full(template_var.shape, np.nan)
                            nan_array = xr.DataArray(nan_data, dims=template_var.dims, attrs=template_var.attrs)
                            data_arrays.append(nan_array)
                            
                    except Exception:
                        # Level exists in header but can't be loaded - create NaN DataArray
                        template_var = template_vars[field_name]
                        nan_data = np.full(template_var.shape, np.nan)
                        nan_array = xr.DataArray(nan_data, dims=template_var.dims, attrs=template_var.attrs)
                        data_arrays.append(nan_array)
                else:
                    # Level doesn't exist in this file - create NaN DataArray
                    template_var = template_vars[field_name]
                    nan_data = np.full(template_var.shape, np.nan)
                    nan_array = xr.DataArray(nan_data, dims=template_var.dims, attrs=template_var.attrs)
                    data_arrays.append(nan_array)
            
            # Use xarray's concat function to properly handle dask arrays
            concatenated_array = xr.concat(data_arrays, dim=self.time_dimension_name)
            
            # Convert back to Variable and update attributes
            concatenated_var = Variable(
                dims=concatenated_array.dims,
                data=concatenated_array.data,
                attrs={
                    **concatenated_array.attrs,
                    'concatenated_files': len(self.plotfile_paths),
                    'time_range': f"{self.times[0]} to {self.times[-1]}",
                    'level': self.level,
                    'missing_levels_filled_with_nan': True,
                }
            )
            
            variables[field_name] = concatenated_var
        
        return variables
    
    def get_attrs(self):
        """Return global attributes."""
        ref_attrs = self.reference_store.get_attrs()
        
        return {
            **ref_attrs,
            'title': f'Multi-time AMReX Dataset Level {self.level}',
            'concatenated_files': len(self.plotfile_paths),
            'time_steps': len(self.times),
            'time_range': f"{self.times[0]} to {self.times[-1]}",
            'plotfile_paths': [str(p) for p in self.plotfile_paths],
            'source_files': [p.name for p in self.plotfile_paths],
        }

class AMReXEntrypoint(BackendEntrypoint):
    """
    Unified AMReX backend entrypoint supporting both single and multiple files.
    """
    
    def open_dataset(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer | AbstractDataStore | list,
        *,
        drop_variables: str | Iterable[str] | None = None,
        level: int = 0,
        time_dimension_name: str = None,
        dimension_names: dict = None,
        pattern: str = "plt_*",
        **kwargs
    ):
        """
        Open AMReX plotfile(s) for a single level.
        
        Parameters
        ----------
        filename_or_obj : str, Path, list of str/Path, or directory path
            - Single plotfile directory: loads one time step
            - List of plotfile directories: concatenates along time dimension  
            - Directory containing plotfiles: auto-discovers and concatenates time series
        drop_variables : str or iterable of str, optional
            Variable names to exclude from the dataset
        level : int, default 0
            AMR level to load (0 = base level)
        time_dimension_name : str, optional
            Name for the time dimension (default: 'ocean_time')
        dimension_names : dict, optional
            Custom dimension names, e.g., {'x': 'longitude', 'y': 'latitude', 'z': 'depth'}
        pattern : str, default "plt_*"
            Glob pattern to match plotfiles when input is a directory containing time series
        **kwargs
            Additional arguments (ignored for compatibility)
            
        Returns
        -------
        xarray.Dataset
            Dataset containing AMR level data with lazy dask arrays
        """
        
        # Normalize input to list of plotfile paths
        plotfile_paths = self._resolve_input_to_plotfiles(filename_or_obj, pattern)
        
        # Single file case - use existing single-level logic
        if len(plotfile_paths) == 1:
            store = AMReXSingleLevelStore(
                plotfile_paths[0], 
                level=level, 
                time_dimension_name=time_dimension_name,
                dimension_names=dimension_names
            )
            
            # Get variables, dimensions, and attributes
            variables = store.get_variables()
            attributes = store.get_attrs()
            
        else:
            # Multiple files case - use multi-time logic
            store = AMReXMultiTimeStore(
                plotfile_paths,
                level=level,
                time_dimension_name=time_dimension_name,
                dimension_names=dimension_names
            )
            
            # Get variables, dimensions, and attributes
            variables = store.get_variables()
            attributes = store.get_attrs()
        
        # Filter out dropped variables
        if drop_variables:
            if isinstance(drop_variables, str):
                drop_variables = [drop_variables]
            for var in drop_variables:
                variables.pop(var, None)
        
        # Create coordinate dict (separate from data variables)
        coords = {}
        data_vars = {}
        
        ### RDH -- this needs a refactor for generalization. Works for current REMORA plot files.
        # Get the time dimension name
        time_dim_name = time_dimension_name or 'ocean_time'
        
        for name, var in variables.items():
            if name in ['x', 'y', 'z', time_dim_name]:
                coords[name] = var
            else:
                data_vars[name] = var
        #### /RDH

        # Create dataset
        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=attributes
        )
    
    def _resolve_input_to_plotfiles(self, filename_or_obj, pattern: str = "plt_*") -> list:
        """
        Resolve various input types to a list of plotfile paths.
        
        Parameters
        ----------
        filename_or_obj : str, Path, list, or other
            Input that could be:
            - Single plotfile directory
            - List of plotfile directories  
            - Directory containing plotfiles
            - Other xarray input types
        pattern : str
            Glob pattern for finding plotfiles in directories
            
        Returns
        -------
        list of Path
            List of plotfile directory paths
        """
        # Handle list/tuple input
        if isinstance(filename_or_obj, (list, tuple)):
            return [Path(p) for p in filename_or_obj]
        
        # Handle non-path inputs (ReadBuffer, AbstractDataStore, etc.)
        if not isinstance(filename_or_obj, (str, os.PathLike)):
            return [filename_or_obj]
        
        # Convert to Path
        path = Path(filename_or_obj)
        
        # If path doesn't exist, return as-is (might be handled elsewhere)
        if not path.exists():
            return [path]
        
        # If it's a file, return as-is
        if path.is_file():
            return [path]
        
        # If it's a directory, we need to determine if it's:
        # 1. A single plotfile directory
        # 2. A directory containing multiple plotfiles
        
        if self._is_amrex_plotfile_directory(path):
            # It's a single plotfile directory
            return [path]
        else:
            # It's a directory containing plotfiles - find them using pattern
            plotfiles = list(path.glob(pattern))
            # Filter to only directories that look like plotfiles
            plotfiles = [p for p in plotfiles if p.is_dir() and self._is_amrex_plotfile_directory(p)]
            
            if not plotfiles:
                # No plotfiles found with pattern, maybe the directory itself is a plotfile
                # or the pattern didn't match - try to treat as single plotfile
                return [path]
            
            # Sort plotfiles by simulation time
            return self._sort_plotfiles_by_time(plotfiles)
    
    def _is_amrex_plotfile_directory(self, path: Path) -> bool:
        """Check if a directory looks like an AMReX plotfile."""
        try:
            if not path.is_dir():
                return False
            
            # Check for AMReX plotfile structure
            header_file = path / 'Header'
            level_0_dir = path / 'Level_0'
            
            if not (header_file.exists() and level_0_dir.is_dir()):
                return False
            
            # Check for Cell_H file in Level_0
            cell_h_file = level_0_dir / 'Cell_H'
            return cell_h_file.exists()
            
        except Exception:
            return False
    
    def _sort_plotfiles_by_time(self, plotfiles: list) -> list:
        """Sort plotfiles by simulation time."""
        try:
            time_file_pairs = []
            
            for pf in plotfiles:
                try:
                    meta = AMReXDatasetMeta(pf)
                    time_file_pairs.append((meta.current_time, pf))
                except Exception:
                    # If we can't read the time, use filename for sorting
                    time_file_pairs.append((float('inf'), pf))
            
            # Sort by time
            time_file_pairs.sort(key=lambda x: x[0])
            return [pair[1] for pair in time_file_pairs]
            
        except Exception:
            # Fallback to filename sorting
            return sorted(plotfiles)
    
    def guess_can_open(self, filename_or_obj: str | os.PathLike[Any] | ReadBuffer | AbstractDataStore) -> bool:
        """Check if this looks like an AMReX plotfile."""
        try:
            path = Path(filename_or_obj)
            if not path.is_dir():
                return False
            
            # Check for AMReX plotfile structure
            header_file = path / 'Header'
            level_0_dir = path / 'Level_0'
            
            if not (header_file.exists() and level_0_dir.is_dir()):
                return False
            
            # Check for Cell_H file in Level_0
            cell_h_file = level_0_dir / 'Cell_H'
            return cell_h_file.exists()
            
        except Exception:
            return False

    open_dataset_parameters = ["filename_or_obj", "drop_variables", "level"]
    
    description = "Single-level AMReX plotfile backend with lazy dask loading"
    url = "https://github.com/your-repo/xamrex"
