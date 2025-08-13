"""
Utilities for working with multi-time AMReX datasets.
Provides functions to open and concatenate multiple time steps following xarray conventions.
"""
from pathlib import Path
from typing import Dict, List, Optional, Union
import xarray as xr
import numpy as np
import glob

def open_amrex_time_series(plotfile_paths: Union[List[Union[str, Path]], str], 
                          level: int = 0,
                          time_dimension_name: str = None,
                          dimension_names: dict = None,
                          **kwargs) -> xr.Dataset:
    """
    Open multiple AMReX plotfiles as a time series dataset following xarray conventions.
    
    Parameters
    ----------
    plotfile_paths : list of str/Path or glob pattern
        List of paths to AMReX plotfile directories, or a glob pattern to match files
    level : int, default 0
        AMR level to load from each plotfile
    time_dimension_name : str, optional
        Name for the time dimension (default: 'ocean_time')
    dimension_names : dict, optional
        Custom dimension names
    **kwargs
        Additional arguments passed to the backend
        
    Returns
    -------
    xarray.Dataset
        Dataset with concatenated time steps
    """
    # Handle glob patterns
    if isinstance(plotfile_paths, str):
        if '*' in plotfile_paths or '?' in plotfile_paths:
            # It's a glob pattern
            plotfile_paths = sorted(glob.glob(plotfile_paths))
            if not plotfile_paths:
                raise ValueError(f"No files found matching pattern")
        else:
            # Single file path
            plotfile_paths = [plotfile_paths]
    
    # Convert to Path objects and validate
    validated_paths = []
    for path in plotfile_paths:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Plotfile not found: {path}")
        if not path_obj.is_dir():
            raise ValueError(f"Plotfile path must be a directory: {path}")
        validated_paths.append(path_obj)
    
    if len(validated_paths) == 0:
        raise ValueError("No valid plotfile paths provided")
    
    # Use the unified backend directly
    from .backend import AMReXEntrypoint
    backend = AMReXEntrypoint()
    return backend.open_dataset(
        validated_paths,
        level=level,
        time_dimension_name=time_dimension_name,
        dimension_names=dimension_names,
        **kwargs
    )

def find_amrex_time_series(directory: Union[str, Path], 
                          pattern: str = "plt_*",
                          sort_by_time: bool = True) -> List[Path]:
    """
    Find AMReX plotfiles in a directory that form a time series.
    
    Parameters
    ----------
    directory : str or Path
        Directory to search for plotfiles
    pattern : str, default "plt_*"
        Glob pattern to match plotfile directories
    sort_by_time : bool, default True
        Whether to sort by simulation time (requires reading metadata)
        
    Returns
    -------
    list of Path
        List of plotfile paths sorted by time
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Find matching directories
    plotfiles = list(directory.glob(pattern))
    plotfiles = [p for p in plotfiles if p.is_dir()]
    
    if not plotfiles:
        return []
    
    if sort_by_time:
        # Read simulation times and sort
        from .AMReX_array import AMReXDatasetMeta
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
        plotfiles = [pair[1] for pair in time_file_pairs]
    else:
        # Sort by filename
        plotfiles.sort()
    
    return plotfiles

def create_time_series_from_directory(directory: Union[str, Path],
                                    pattern: str = "plt_*",
                                    level: int = 0,
                                    time_dimension_name: str = None,
                                    dimension_names: dict = None,
                                    **kwargs) -> xr.Dataset:
    """
    Create a time series dataset from all matching plotfiles in a directory.
    
    Parameters
    ----------
    directory : str or Path
        Directory containing AMReX plotfiles
    pattern : str, default "plt_*"
        Glob pattern to match plotfile directories
    level : int, default 0
        AMR level to load
    time_dimension_name : str, optional
        Name for the time dimension
    dimension_names : dict, optional
        Custom dimension names
    **kwargs
        Additional arguments passed to open_amrex_time_series
        
    Returns
    -------
    xarray.Dataset
        Multi-time dataset
    """
    plotfiles = find_amrex_time_series(directory, pattern, sort_by_time=True)
    
    if not plotfiles:
        raise ValueError(f"No plotfiles found in {directory} matching pattern '{pattern}'")
    
    return open_amrex_time_series(
        plotfiles,
        level=level,
        time_dimension_name=time_dimension_name,
        dimension_names=dimension_names,
        **kwargs
    )

def validate_time_series_compatibility(plotfile_paths: List[Union[str, Path]], 
                                     level: int = 0) -> Dict[str, any]:
    """
    Validate that a list of plotfiles can be concatenated into a time series.
    
    Parameters
    ----------
    plotfile_paths : list of str or Path
        List of plotfile paths to validate
    level : int, default 0
        AMR level to check
        
    Returns
    -------
    dict
        Validation results with compatibility information
    """
    from .AMReX_array import AMReXDatasetMeta
    
    results = {
        'compatible': True,
        'issues': [],
        'file_count': len(plotfile_paths),
        'time_range': None,
        'fields': None,
        'domain_info': None,
    }
    
    if len(plotfile_paths) == 0:
        results['compatible'] = False
        results['issues'].append("No plotfiles provided")
        return results
    
    metas = []
    times = []
    
    # Read metadata from all files
    for i, path in enumerate(plotfile_paths):
        try:
            meta = AMReXDatasetMeta(path)
            metas.append(meta)
            times.append(meta.current_time)
        except Exception as e:
            results['compatible'] = False
            results['issues'].append(f"Failed to read metadata from file {i}: {e}")
            continue
    
    if not metas:
        results['compatible'] = False
        results['issues'].append("No valid metadata found")
        return results
    
    # Check compatibility
    ref_meta = metas[0]
    results['time_range'] = (min(times), max(times))
    results['fields'] = ref_meta.field_list
    results['domain_info'] = {
        'dimensionality': ref_meta.dimensionality,
        'left_edge': ref_meta.domain_left_edge.tolist(),
        'right_edge': ref_meta.domain_right_edge.tolist(),
    }
    
    for i, meta in enumerate(metas[1:], 1):
        # Check dimensionality
        if meta.dimensionality != ref_meta.dimensionality:
            results['compatible'] = False
            results['issues'].append(f"File {i}: dimensionality mismatch ({meta.dimensionality} vs {ref_meta.dimensionality})")
        
        # Check fields
        if set(meta.field_list) != set(ref_meta.field_list):
            results['compatible'] = False
            results['issues'].append(f"File {i}: field list mismatch")
        
        # Check domain
        if not np.allclose(meta.domain_left_edge, ref_meta.domain_left_edge):
            results['compatible'] = False
            results['issues'].append(f"File {i}: domain left edge mismatch")
        
        if not np.allclose(meta.domain_right_edge, ref_meta.domain_right_edge):
            results['compatible'] = False
            results['issues'].append(f"File {i}: domain right edge mismatch")
        
        # Check level availability
        if meta.max_level < level:
            results['compatible'] = False
            results['issues'].append(f"File {i}: level {level} not available (max: {meta.max_level})")
    
    # Check for duplicate times
    if len(set(times)) != len(times):
        results['issues'].append("Warning: duplicate time values found")
    
    return results

def extract_time_slice(dataset: xr.Dataset, 
                      time_range: tuple = None,
                      time_indices: slice = None) -> xr.Dataset:
    """
    Extract a time slice from a multi-time AMReX dataset.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Multi-time dataset
    time_range : tuple of float, optional
        (start_time, end_time) to extract
    time_indices : slice, optional
        Index-based slice to extract
        
    Returns
    -------
    xarray.Dataset
        Time-sliced dataset
    """
    time_dim = None
    for dim in dataset.dims:
        if 'time' in dim.lower():
            time_dim = dim
            break
    
    if time_dim is None:
        raise ValueError("No time dimension found in dataset")
    
    if time_range is not None:
        start_time, end_time = time_range
        time_coord = dataset.coords[time_dim]
        mask = (time_coord >= start_time) & (time_coord <= end_time)
        return dataset.isel({time_dim: mask})
    
    elif time_indices is not None:
        return dataset.isel({time_dim: time_indices})
    
    else:
        raise ValueError("Either time_range or time_indices must be provided")

def compute_time_statistics(dataset: xr.Dataset, 
                          variables: Union[str, List[str]] = None,
                          statistics: List[str] = None) -> xr.Dataset:
    """
    Compute time-based statistics for multi-time AMReX dataset.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Multi-time dataset
    variables : str or list of str, optional
        Variables to compute statistics for (default: all data variables)
    statistics : list of str, optional
        Statistics to compute (default: ['mean', 'std', 'min', 'max'])
        
    Returns
    -------
    xarray.Dataset
        Dataset with computed statistics
    """
    if statistics is None:
        statistics = ['mean', 'std', 'min', 'max']
    
    if variables is None:
        variables = list(dataset.data_vars.keys())
    elif isinstance(variables, str):
        variables = [variables]
    
    # Find time dimension
    time_dim = None
    for dim in dataset.dims:
        if 'time' in dim.lower():
            time_dim = dim
            break
    
    if time_dim is None:
        raise ValueError("No time dimension found in dataset")
    
    result_vars = {}
    
    for var_name in variables:
        if var_name not in dataset.data_vars:
            continue
        
        var = dataset[var_name]
        
        for stat in statistics:
            if stat == 'mean':
                result_vars[f'{var_name}_mean'] = var.mean(dim=time_dim)
            elif stat == 'std':
                result_vars[f'{var_name}_std'] = var.std(dim=time_dim)
            elif stat == 'min':
                result_vars[f'{var_name}_min'] = var.min(dim=time_dim)
            elif stat == 'max':
                result_vars[f'{var_name}_max'] = var.max(dim=time_dim)
            elif stat == 'median':
                result_vars[f'{var_name}_median'] = var.median(dim=time_dim)
    
    # Create result dataset with spatial coordinates only
    coords = {name: coord for name, coord in dataset.coords.items() 
              if name != time_dim}
    
    return xr.Dataset(
        data_vars=result_vars,
        coords=coords,
        attrs={
            **dataset.attrs,
            'title': f"Time statistics of {dataset.attrs.get('title', 'AMReX dataset')}",
            'statistics_computed': statistics,
            'time_range': f"Based on {dataset.dims[time_dim]} time steps",
        }
    )
