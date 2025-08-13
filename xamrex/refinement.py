"""
Utilities for handling AMReX adaptive mesh refinement (AMR) logic.
Centralizes refinement factor calculations and coordinate transformations.
"""
import numpy as np
from typing import Tuple, List


class RefinementHandler:
    """
    Handles AMReX refinement calculations and coordinate transformations.
    
    Centralizes the complex logic around:
    - Refinement factor calculations
    - Coordinate generation for different levels
    - Dimension-specific refinement (x,y refined, z not refined)
    """
    
    def __init__(self, base_refinement: int, dimensionality: int, domain_dimensions: np.ndarray):
        """
        Initialize refinement handler.
        
        Parameters
        ----------
        base_refinement : int
            Base refinement factor (typically 2)
        dimensionality : int
            Spatial dimensionality (2 or 3)
        domain_dimensions : np.ndarray
            Base level domain dimensions
        """
        self.base_refinement = base_refinement
        self.dimensionality = dimensionality
        self.domain_dimensions = domain_dimensions
    
    def get_refinement_factors(self, level: int) -> np.ndarray:
        """
        Get refinement factors for a specific level.
        
        In AMReX convention:
        - x,y dimensions are refined by base_refinement^level
        - z dimension is not refined (factor = 1)
        
        Parameters
        ----------
        level : int
            AMR level
            
        Returns
        -------
        np.ndarray
            Refinement factors for each dimension
        """
        if self.dimensionality == 3:
            # x,y get refinement, z stays at 1
            return np.array([
                self.base_refinement ** level,  # x
                self.base_refinement ** level,  # y  
                1                               # z
            ])
        else:
            # 2D: x,y get refinement
            return np.array([
                self.base_refinement ** level,  # x
                self.base_refinement ** level   # y
            ])
    
    def get_refined_dimensions(self, level: int) -> np.ndarray:
        """
        Get grid dimensions for a specific level.
        
        Parameters
        ----------
        level : int
            AMR level
            
        Returns
        -------
        np.ndarray
            Grid dimensions for this level
        """
        refinement_factors = self.get_refinement_factors(level)
        base_dims = self.domain_dimensions[:self.dimensionality]
        return base_dims * refinement_factors
    
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
            spatial_shape = tuple(refined_dims)
        
        if include_time:
            return (1,) + spatial_shape
        else:
            return spatial_shape
    
    def get_coordinate_arrays(self, level: int, domain_left_edge: np.ndarray, 
                            domain_right_edge: np.ndarray) -> dict:
        """
        Generate coordinate arrays for a specific level.
        
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
        refinement_factors = self.get_refinement_factors(level)
        refined_dims = self.get_refined_dimensions(level)
        
        domain_extent = domain_right_edge - domain_left_edge
        base_dims = self.domain_dimensions[:self.dimensionality]
        base_grid_spacing = domain_extent / base_dims
        
        coords = {}
        dim_names = ['x', 'y', 'z'][:self.dimensionality]
        
        for i, dim in enumerate(dim_names):
            coord_start = domain_left_edge[i]
            
            # Calculate spacing for this level and dimension
            base_spacing = float(base_grid_spacing[i])
            refinement_factor_for_dim = refinement_factors[i]
            spacing = base_spacing / refinement_factor_for_dim
            
            n_points = refined_dims[i]
            
            # Cell-centered coordinates
            coord_array = coord_start + (np.arange(n_points) + 0.5) * spacing
            coords[dim] = coord_array
        
        return coords
    
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
            # Handle potential z-refinement mismatch
            if level > 0:
                refinement_factor = self.base_refinement ** level
                
                # Check if z indices indicate refinement in the FAB data
                if hi_k > self.domain_dimensions[2]:
                    # FAB data is refined in z, map back to unrefined z coordinates
                    lo_k_unrefined = lo_k // refinement_factor
                    hi_k_unrefined = hi_k // refinement_factor
                    return (slice(lo_k_unrefined, hi_k_unrefined),
                           slice(lo_j, hi_j),
                           slice(lo_i, hi_i))
            
            # Normal case: (z, y, x) in Fortran order
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
    base_refinement = int(meta.ref_factors[0]) if len(meta.ref_factors) > 0 else 2
    return RefinementHandler(
        base_refinement=base_refinement,
        dimensionality=meta.dimensionality,
        domain_dimensions=meta.domain_dimensions
    )
