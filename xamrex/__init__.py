"""
xamrex - AMReX plotfile reader with C-grid support

This package provides an xarray backend for reading AMReX plotfiles with:
- Automatic detection of staggered grid variables (rho, u, v, w, psi points)
- Support for 2D and 3D variables
- Multi-file time series with level masking
- Multiple AMR levels with proper coordinate scaling
- xgcm-compatible grid metadata
"""

from .backend import AMReXCGridEntrypoint

__version__ = "2.0.0"
__all__ = ["AMReXCGridEntrypoint"]
