"""
Configuration utilities for xamrex package.
Centralizes configuration options and provides default values.
"""
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class XamrexConfig:
    """
    Configuration for xamrex package behavior.
    
    Attributes
    ----------
    time_dimension_name : str
        Default name for time dimension
    default_dimension_names : dict
        Default spatial dimension names
    fill_value : float
        Default fill value for missing data
    chunk_size_mb : int
        Default chunk size for dask arrays in MB
    """
    time_dimension_name: str = 'ocean_time'
    default_dimension_names: Dict[str, str] = None
    fill_value: float = float('nan')
    chunk_size_mb: int = 64
    
    def __post_init__(self):
        if self.default_dimension_names is None:
            self.default_dimension_names = {
                'x': 'x',
                'y': 'y', 
                'z': 'z'
            }

# Global configuration instance
_config = XamrexConfig()

def get_config() -> XamrexConfig:
    """Get the current configuration."""
    return _config

def set_config(**kwargs) -> None:
    """
    Update configuration values.
    
    Parameters
    ----------
    **kwargs
        Configuration parameters to update
    """
    global _config
    for key, value in kwargs.items():
        if hasattr(_config, key):
            setattr(_config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")

def reset_config() -> None:
    """Reset configuration to defaults."""
    global _config
    _config = XamrexConfig()

def get_dimension_names(custom_names: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Get dimension names, with custom names taking precedence.
    
    Parameters
    ----------
    custom_names : dict, optional
        Custom dimension names to override defaults
        
    Returns
    -------
    dict
        Final dimension names to use
    """
    names = _config.default_dimension_names.copy()
    if custom_names:
        names.update(custom_names)
    return names

def get_time_dimension_name(custom_name: Optional[str] = None) -> str:
    """
    Get time dimension name.
    
    Parameters
    ----------
    custom_name : str, optional
        Custom time dimension name
        
    Returns
    -------
    str
        Time dimension name to use
    """
    return custom_name or _config.time_dimension_name
