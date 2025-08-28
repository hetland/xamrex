"""
Utilities for handling AMReX adaptive mesh refinement (AMR) logic.
Centralizes refinement factor calculations and coordinate transformations.
"""
import numpy as np
from typing import Tuple, List


class RefinementHandler:
    """
    Handles AMReX refinement calculations and coordinate transformations.
    
    Now uses actual header data instead of mathematical estimates.
    """
    
    def __init__(self, meta):
        """
        Initialize refinement handler with metadata.
        
        Parameters
        ----------
        meta : AMReXDatasetMeta
            Metadata object containing parsed header information
        """
        self.meta = meta
        self.base_refinement = int(meta.ref_factors[0]) if len(meta.ref_factors) > 0 else 2
        self.dimensionality = meta.dimensionality
    
    def get_refinement_factors(self, level: int) -> np.ndarray:
        """
        Get refinement factors for a specific level from header data.
        
        Parameters
        ----------
        level : int
            AMR level
            
        Returns
        -------
        np.ndarray
            Refinement factors for each dimension
        """
        return np.array(self.meta.get_level_refinement_factors(level))
    
    def get_refined_dimensions(self, level: int) -> np.ndarray:
        """
        Get grid dimensions for a specific level from header data.
        
        Parameters
        ----------
        level : int
            AMR level
            
        Returns
        -------
        np.ndarray
            Grid dimensions for this level
        """

        print(np.array(self.meta.get_level_dimensions(level)))
        return np.array(self.meta.get_level_dimensions(level))
    
    def get_full_shape(self, level: int, include_time: bool = True) -> Tuple[int, ...]:
        """
        Get full array shape for a specific level.
        
        Parameters
        ----------
        level : int
            AMR level
        include_time : bool, default True
            Whether to include singleton time dimension
            
        Returns
        -------
        tuple
            Array shape (time, z, y, x) or (z, y, x) or (y, x)
        """
        refined_dims = self.get_refined_dimensions(level)
        
        if self.dimensionality == 3:
            # Fortran order: z, y, x
            spatial_shape = tuple(refined_dims[::-1])
        else:
            # 2D: y, x
            spatial_shape = tuple(refined_dims[::-1])
        
        if include_time:
            return (1,) + spatial_shape
        else:
            return spatial_shape
    
    def get_coordinate_arrays(self, level: int, domain_left_edge: np.ndarray, 
                            domain_right_edge: np.ndarray) -> dict:
        """
        Generate coordinate arrays for a specific level using header data.
        
        Parameters
        ----------
        level : int
            AMR level
        domain_left_edge : np.ndarray
            Domain left edge coordinates
        domain_right_edge : np.ndarray
            Domain right edge coordinates
            
        Returns
        -------
        dict
            Dictionary of coordinate arrays {dim_name: coord_array}
        """
        # Use the metadata method directly
        return self.meta.get_level_coordinate_arrays(level)
    
    def validate_fab_indices(self, level: int, lo_indices: np.ndarray, 
                           hi_indices: np.ndarray) -> bool:
        """
        Validate that FAB indices are within expected bounds for a level.
        
        Parameters
        ----------
        level : int
            AMR level
        lo_indices : np.ndarray
            Low indices (i, j, k)
        hi_indices : np.ndarray  
            High indices (i, j, k)
            
        Returns
        -------
        bool
            True if indices are valid
        """
        refined_dims = self.get_refined_dimensions(level)
        
        # Check bounds
        if np.any(lo_indices < 0):
            return False
        
        if np.any(hi_indices > refined_dims):
            return False
            
        if np.any(lo_indices >= hi_indices):
            return False
            
        return True
    
    def map_fab_to_global_indices(self, level: int, lo_i: int, lo_j: int, lo_k: int,
                                 hi_i: int, hi_j: int, hi_k: int) -> Tuple[slice, ...]:
        """
        Map FAB indices to global grid indices for data placement.
        
        Parameters
        ----------
        level : int
            AMR level
        lo_i, lo_j, lo_k : int
            FAB low indices
        hi_i, hi_j, hi_k : int
            FAB high indices
            
        Returns
        -------
        tuple of slice
            Slices for placing FAB data in global grid
        """
        if self.dimensionality == 3:
            # Direct mapping: (z, y, x) in Fortran order
            return (slice(lo_k, hi_k), slice(lo_j, hi_j), slice(lo_i, hi_i))
        else:
            # 2D case: (y, x)
            return (slice(lo_j, hi_j), slice(lo_i, hi_i))


def create_refinement_handler(meta) -> RefinementHandler:
    """
    Create a RefinementHandler from AMReX metadata.
    
    Parameters
    ----------
    meta : AMReXDatasetMeta
        Metadata object
        
    Returns
    -------
    RefinementHandler
        Configured refinement handler
    """
    return RefinementHandler(meta)
