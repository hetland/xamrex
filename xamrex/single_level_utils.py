"""
Utilities for working with single-level AMReX datasets.
Provides functions to load and work with multiple levels as separate datasets.
"""
from pathlib import Path
from typing import Dict, List, Optional, Union
import xarray as xr
import numpy as np


def open_amrex_levels(plotfile_path: Union[str, Path], 
                     levels: Union[str, int, List[int]] = 'all',
                     **kwargs) -> Dict[int, xr.Dataset]:
    """
    Open multiple AMR levels as separate datasets.
    
    Parameters
    ----------
    plotfile_path : str or Path
        Path to AMReX plotfile directory
    levels : str, int, or list of int, default 'all'
        Which levels to load:
        - 'all': Load all available levels
        - int: Load single level
        - list: Load specific levels
    **kwargs
        Additional arguments passed to xr.open_dataset
        
    Returns
    -------
    dict
        Dictionary mapping level number to xarray.Dataset
    """
    plotfile_path = Path(plotfile_path)
    
    # Determine available levels
    available_levels = get_available_levels_from_file(plotfile_path)
    
    # Parse level selection
    if levels == 'all':
        levels_to_load = available_levels
    elif isinstance(levels, int):
        levels_to_load = [levels]
    elif isinstance(levels, (list, tuple)):
        levels_to_load = list(levels)
    else:
        raise ValueError(f"Invalid levels specification: {levels}")
    
    # Validate requested levels
    for level in levels_to_load:
        if level not in available_levels:
            raise ValueError(f"Level {level} not available. Available: {available_levels}")
    
    # Load each level as separate dataset
    datasets = {}
    for level in levels_to_load:
        datasets[level] = xr.open_dataset(
            plotfile_path, 
            engine="amrex", 
            level=level,
            **kwargs
        )
    
    return datasets


def get_available_levels_from_file(plotfile_path: Union[str, Path]) -> List[int]:
    """
    Get available AMR levels from plotfile without loading data.
    
    Parameters
    ----------
    plotfile_path : str or Path
        Path to AMReX plotfile directory
        
    Returns
    -------
    list of int
        Available level numbers
    """
    from .AMReX_array import AMReXDatasetMeta
    
    meta = AMReXDatasetMeta(plotfile_path)
    return list(range(meta.max_level + 1))


def compare_level_resolutions(datasets: Dict[int, xr.Dataset]) -> Dict[int, Dict[str, float]]:
    """
    Compare grid resolutions across levels.
    
    Parameters
    ----------
    datasets : dict
        Dictionary of level -> dataset from open_amrex_levels
        
    Returns
    -------
    dict
        Nested dict: level -> {dimension -> spacing}
    """
    resolutions = {}
    
    for level, ds in datasets.items():
        level_res = {}
        for dim in ['x', 'y', 'z']:
            if dim in ds.coords:
                coord = ds.coords[dim].values
                if len(coord) > 1:
                    spacing = coord[1] - coord[0]
                    level_res[dim] = spacing
        resolutions[level] = level_res
    
    return resolutions


def get_refinement_factors(datasets: Dict[int, xr.Dataset]) -> Dict[int, int]:
    """
    Calculate refinement factors relative to level 0.
    
    Parameters
    ----------
    datasets : dict
        Dictionary of level -> dataset from open_amrex_levels
        
    Returns
    -------
    dict
        Level -> refinement factor (relative to level 0)
    """
    if 0 not in datasets:
        raise ValueError("Level 0 must be present to calculate refinement factors")
    
    base_res = compare_level_resolutions({0: datasets[0]})[0]
    all_res = compare_level_resolutions(datasets)
    
    factors = {}
    for level in datasets.keys():
        if 'x' in base_res and 'x' in all_res[level]:
            factor = base_res['x'] / all_res[level]['x']
            factors[level] = int(round(factor))
        else:
            factors[level] = 1
    
    return factors


def extract_common_region(datasets: Dict[int, xr.Dataset], 
                         field: str,
                         region: Optional[Dict[str, slice]] = None) -> Dict[int, xr.DataArray]:
    """
    Extract the same physical region from multiple levels.
    
    Parameters
    ----------
    datasets : dict
        Dictionary of level -> dataset from open_amrex_levels
    field : str
        Field name to extract
    region : dict, optional
        Region specification as {dim: slice(start, stop)}
        If None, uses overlapping region of all levels
        
    Returns
    -------
    dict
        Level -> DataArray for the specified region
    """
    if region is None:
        # Find overlapping region
        region = find_overlapping_region(datasets)
    
    extracted = {}
    for level, ds in datasets.items():
        if field not in ds.data_vars:
            continue
            
        var = ds[field]
        
        # Convert physical coordinates to indices for this level
        selection = {}
        for dim, coord_slice in region.items():
            if dim in var.dims:
                coord = ds.coords[dim].values
                
                # Find indices corresponding to physical coordinates
                start_idx = np.searchsorted(coord, coord_slice.start)
                stop_idx = np.searchsorted(coord, coord_slice.stop)
                
                selection[dim] = slice(start_idx, stop_idx)
        
        extracted[level] = var.isel(selection)
    
    return extracted


def find_overlapping_region(datasets: Dict[int, xr.Dataset]) -> Dict[str, slice]:
    """
    Find the physical region that overlaps across all levels.
    
    Parameters
    ----------
    datasets : dict
        Dictionary of level -> dataset from open_amrex_levels
        
    Returns
    -------
    dict
        Region specification as {dim: slice(start, stop)}
    """
    overlapping_region = {}
    
    for dim in ['x', 'y', 'z']:
        if all(dim in ds.coords for ds in datasets.values()):
            # Find min and max bounds across all levels
            min_vals = []
            max_vals = []
            
            for ds in datasets.values():
                coord = ds.coords[dim].values
                min_vals.append(coord.min())
                max_vals.append(coord.max())
            
            # Use the most restrictive bounds (largest min, smallest max)
            region_start = max(min_vals)
            region_stop = min(max_vals)
            
            overlapping_region[dim] = slice(region_start, region_stop)
    
    return overlapping_region




def create_level_summary(datasets: Dict[int, xr.Dataset]) -> xr.Dataset:
    """
    Create summary information about AMR levels.
    
    Parameters
    ----------
    datasets : dict
        Dictionary of level -> dataset from open_amrex_levels
        
    Returns
    -------
    xarray.Dataset
        Summary dataset with level information
    """
    levels = sorted(datasets.keys())
    
    # Collect information
    resolutions = compare_level_resolutions(datasets)
    refinement_factors = get_refinement_factors(datasets)
    
    summary_data = {}
    
    # Grid spacing for each dimension
    for dim in ['x', 'y', 'z']:
        if all(dim in resolutions[level] for level in levels):
            spacings = [resolutions[level][dim] for level in levels]
            summary_data[f'{dim}_spacing'] = (['level'], spacings)
    
    # Refinement factors
    factors = [refinement_factors[level] for level in levels]
    summary_data['refinement_factor'] = (['level'], factors)
    
    # Grid sizes
    for dim in ['x', 'y', 'z']:
        if all(dim in datasets[level].dims for level in levels):
            sizes = [datasets[level].dims[dim] for level in levels]
            summary_data[f'{dim}_size'] = (['level'], sizes)
    
    # Coverage (for levels with masking)
    if len(datasets) > 0:
        sample_ds = next(iter(datasets.values()))
        data_vars = [var for var in sample_ds.data_vars if var not in ['x', 'y', 'z']]
        
        if data_vars:
            sample_field = data_vars[0]
            coverage_values = []
            
            for level in levels:
                if sample_field in datasets[level].data_vars:
                    var = datasets[level][sample_field]
                    if hasattr(var.values, 'mask'):
                        total_points = var.size
                        valid_points = np.sum(~var.values.mask)
                        coverage = valid_points / total_points
                    else:
                        coverage = 1.0
                    coverage_values.append(coverage)
                else:
                    coverage_values.append(0.0)
            
            summary_data['coverage_fraction'] = (['level'], coverage_values)
    
    return xr.Dataset(
        data_vars=summary_data,
        coords={'level': (['level'], levels)},
        attrs={
            'title': 'AMR Level Summary',
            'description': 'Summary of AMR refinement levels',
            'total_levels': len(levels),
            'max_level': max(levels),
        }
    )


# Convenience functions
def load_level(plotfile_path: Union[str, Path], level: int = 0, **kwargs) -> xr.Dataset:
    """
    Load a single AMR level.
    
    Convenience function equivalent to:
    xr.open_dataset(plotfile_path, engine='amrex', level=level)
    """
    return xr.open_dataset(plotfile_path, engine='amrex', level=level, **kwargs)


def load_base_level(plotfile_path: Union[str, Path], **kwargs) -> xr.Dataset:
    """Load the base level (level 0) of an AMReX plotfile."""
    return load_level(plotfile_path, level=0, **kwargs)


def get_max_level(plotfile_path: Union[str, Path]) -> int:
    """Get the maximum available refinement level."""
    available_levels = get_available_levels_from_file(plotfile_path)
    return max(available_levels) if available_levels else 0


def calculate_effective_resolution(datasets: Dict[int, xr.Dataset], 
                                 field: str) -> xr.Dataset:
    """
    Calculate effective resolution by combining information from all levels.
    Returns a dataset showing the finest available resolution at each point.
    
    Parameters
    ----------
    datasets : dict
        Dictionary of level -> dataset from open_amrex_levels
    field : str
        Field name to analyze
        
    Returns
    -------
    xarray.Dataset
        Dataset with effective resolution information
    """
    # Use finest level as base grid
    finest_level = max(datasets.keys())
    base_ds = datasets[finest_level]
    
    if field not in base_ds.data_vars:
        raise ValueError(f"Field {field} not found in finest level dataset")
    
    # Create resolution map (refinement level at each point)
    resolution_map = xr.full_like(base_ds[field], fill_value=-1, dtype=int)
    
    # Fill in resolution levels, starting from coarsest
    for level in sorted(datasets.keys()):
        ds = datasets[level]
        if field in ds.data_vars:
            var = ds[field]
            
            # Find where this level has valid data
            if hasattr(var.values, 'mask'):
                valid_mask = ~var.values.mask
            else:
                valid_mask = ~np.isnan(var.values)
            
            # Map to finest grid coordinates
            # This is a simplified version - full implementation would need
            # proper coordinate transformation and interpolation
            if level == finest_level:
                resolution_map = resolution_map.where(~valid_mask, level)
            # TODO: Add coordinate mapping for other levels
    
    return xr.Dataset({
        f'{field}_effective_level': resolution_map,
        f'{field}_data': base_ds[field]
    })
