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
        Get whole grid dimensions for a specific refinement level from header data.
        
        Parameters
        ----------
        level : int
            AMR level
            
        Returns
        -------
        np.ndarray
            Grid dimensions for this level
        """

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
