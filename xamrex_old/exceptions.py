"""
Custom exceptions for xamrex package.
Provides clear error messages for common issues.
"""

class XamrexError(Exception):
    """Base exception for xamrex package."""
    pass

class PlotfileError(XamrexError):
    """Raised when there are issues with AMReX plotfile structure or content."""
    pass

class LevelError(XamrexError):
    """Raised when requested AMR level is not available."""
    pass

class CompatibilityError(XamrexError):
    """Raised when plotfiles are not compatible for concatenation."""
    pass

class RefinementError(XamrexError):
    """Raised when there are issues with AMR refinement calculations.""" 
    pass

def validate_plotfile_path(path):
    """
    Validate that a path points to a valid AMReX plotfile.
    
    Parameters
    ----------
    path : Path
        Path to validate
        
    Raises
    ------
    PlotfileError
        If path is not a valid plotfile
    """
    if not path.exists():
        raise PlotfileError(f"Plotfile path does not exist: {path}")
    
    if not path.is_dir():
        raise PlotfileError(f"Plotfile path must be a directory: {path}")
    
    header_file = path / 'Header'
    if not header_file.exists():
        raise PlotfileError(f"Header file not found: {header_file}")
    
    level_0_dir = path / 'Level_0'
    if not level_0_dir.is_dir():
        raise PlotfileError(f"Level_0 directory not found: {level_0_dir}")

def validate_level_exists(meta, level):
    """
    Validate that a level exists in the metadata.
    
    Parameters
    ----------
    meta : AMReXDatasetMeta
        Metadata object
    level : int
        Level to validate
        
    Raises
    ------
    LevelError
        If level is not available
    """
    if level < 0:
        raise LevelError(f"Level must be non-negative, got: {level}")
    
    if level > meta.max_level:
        raise LevelError(
            f"Level {level} not available. Maximum level: {meta.max_level}"
        )
