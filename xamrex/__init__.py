# This file makes xamrex a Python package and enables backend discovery by xarray.

from .backend import AMReXEntrypoint
from . import single_level_utils
from . import multi_time_utils

__version__ = "0.5.0"

# Make key functions easily accessible from single_level_utils
from .single_level_utils import (
    open_amrex_levels,
    get_available_levels_from_file,
    compare_level_resolutions,
    get_refinement_factors,
    extract_common_region,
    create_level_summary,
    load_level,
    load_base_level,
    get_max_level,
    find_overlapping_region,
    calculate_effective_resolution,
)

# Make key functions easily accessible from multi_time_utils
from .multi_time_utils import (
    open_amrex_time_series,
    find_amrex_time_series,
    create_time_series_from_directory,
    validate_time_series_compatibility,
    extract_time_slice,
    compute_time_statistics,
)

__all__ = [
    # Backend classes
    'AMReXEntrypoint',
    'single_level_utils',
    'multi_time_utils',
    
    # Single-level utilities
    'open_amrex_levels',
    'get_available_levels_from_file',
    'compare_level_resolutions',
    'get_refinement_factors',
    'extract_common_region',
    'create_level_summary',
    'load_level',
    'load_base_level',
    'get_max_level',
    'find_overlapping_region',
    'calculate_effective_resolution',
    
    # Multi-time utilities (recommended API for time series)
    'open_amrex_time_series',
    'find_amrex_time_series',
    'create_time_series_from_directory',
    'validate_time_series_compatibility',
    'extract_time_slice',
    'compute_time_statistics',
]
