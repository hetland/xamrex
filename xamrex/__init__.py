# This file makes xamrex a Python package and enables backend discovery by xarray.

from .backend import AMReXEntrypoint
from . import single_level_utils
from . import multi_time_utils

__version__ = "0.5.0"

# Core functions for most common use cases
from .single_level_utils import (
    load_level,
    load_base_level,
    open_amrex_levels,
    get_max_level,
)

from .multi_time_utils import (
    open_amrex_time_series,
    find_amrex_time_series,
    create_time_series_from_directory,
)

__all__ = [
    # Backend
    'AMReXEntrypoint',
    
    # Submodules (for advanced usage)
    'single_level_utils',
    'multi_time_utils',
    
    # Core functions (most commonly used)
    'load_level',
    'load_base_level', 
    'open_amrex_levels',
    'get_max_level',
    'open_amrex_time_series',
    'find_amrex_time_series',
    'create_time_series_from_directory',
]
