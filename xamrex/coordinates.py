"""
C-grid coordinate generation for staggered variables.
"""
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

class CGridCoordinateGenerator:
    """
    Generate coordinates for C-grid staggered variables.
    
    Handles coordinate naming (x_rho, x_u, etc.) and proper staggering offsets.
    """
    
    def __init__(self, domain_left: np.ndarray, domain_right: np.ndarray, 
                 level_dimensions: Dict[int, list], dimensionality: int,
                 refinement_factors: np.ndarray = None, level: int = 0):
        """
        Initialize coordinate generator.
        
        Parameters
        ----------
        domain_left : np.ndarray
            Left edge of domain [x_min, y_min, z_min]
        domain_right : np.ndarray
            Right edge of domain [x_max, y_max, z_max]
        level_dimensions : dict
            {level: [nx, ny, nz]} for all levels
        dimensionality : int
            2 or 3
        refinement_factors : np.ndarray, optional
            Refinement factors for each level
        level : int, default 0
            Current AMR level being processed
        """
        self.domain_left = domain_left
        self.domain_right = domain_right
        self.level_dimensions = level_dimensions
        self.dimensionality = dimensionality
        self.refinement_factors = refinement_factors
        self.level = level
        
        # Calculate refinement ratio for this level
        if refinement_factors is not None and len(refinement_factors) > 0 and level > 0:
            # Refinement factor is cumulative: level 1 = ref[0], level 2 = ref[0]*ref[1], etc.
            self.ref_ratio = int(refinement_factors[0]) ** level
        else:
            self.ref_ratio = 1
    
    def generate_xy_coordinates(self, grid_type: str, 
                               grid_dimensions: tuple) -> Dict[str, Tuple[str, np.ndarray, Dict]]:
        """
        Generate only horizontal (x, y) coordinate arrays with C-grid naming.
        
        Z-coordinates are generated separately to avoid conflicts.
        
        Parameters
        ----------
        grid_type : str
            One of 'rho', 'u', 'v', 'w', 'psi'
        grid_dimensions : tuple
            Grid-specific dimensions (nx, ny) or (nx, ny, nz) from FAB headers
        
        Returns
        -------
        dict
            {coord_name: (dim_name, array, attributes)}
            Example: {'x_rho': ('x_rho', array([...]), {'axis': 'X', ...})}
        """
        coords = {}
        
        # Extract x and y dimensions (first two elements)
        nx, ny = grid_dimensions[0], grid_dimensions[1]
        
        # Generate horizontal coordinates
        x_coord = self._generate_x_coordinate(nx, grid_type)
        y_coord = self._generate_y_coordinate(ny, grid_type)
        
        x_name = f'x_{grid_type}'
        y_name = f'y_{grid_type}'
        
        coords[x_name] = (
            x_name,
            x_coord,
            self._get_coordinate_attrs(x_name, 'X', grid_type)
        )
        
        coords[y_name] = (
            y_name,
            y_coord,
            self._get_coordinate_attrs(y_name, 'Y', grid_type)
        )
        
        return coords
    
    def _generate_x_coordinate(self, nx: int, grid_type: str) -> np.ndarray:
        """
        Generate x-coordinate with appropriate staggering.
        
        Coordinates are based on domain extent and number of rho-cells at this level.
        
        Parameters
        ----------
        nx : int
            Number of points in x direction (already includes staggering from FAB header)
        grid_type : str
            Grid type (rho, u, v, w, psi)
            
        Returns
        -------
        np.ndarray
            X-coordinate array
        """
        x_min = self.domain_left[0]
        x_max = self.domain_right[0]
        
        # Get number of rho-cells at current level
        nx_rho = self.level_dimensions[self.level][0]
        
        # Cell size based on domain extent and rho-cells
        dx = (x_max - x_min) / nx_rho
        
        if grid_type in ['u', 'psi']:
            # Face/corner points - coordinates at cell faces
            # nx = nx_rho + 1 for faces
            x = x_min + np.arange(nx) * dx
        else:
            # Cell centers (rho, v, w)
            # nx = nx_rho for cell centers
            x = x_min + (np.arange(nx) + 0.5) * dx
        
        return x
    
    def _generate_y_coordinate(self, ny: int, grid_type: str) -> np.ndarray:
        """
        Generate y-coordinate with appropriate staggering.
        
        Coordinates are based on domain extent and number of rho-cells at this level.
        
        Parameters
        ----------
        ny : int
            Number of points in y direction (already includes staggering from FAB header)
        grid_type : str
            Grid type (rho, u, v, w, psi)
            
        Returns
        -------
        np.ndarray
            Y-coordinate array
        """
        y_min = self.domain_left[1]
        y_max = self.domain_right[1]
        
        # Get number of rho-cells at current level
        ny_rho = self.level_dimensions[self.level][1]
        
        # Cell size based on domain extent and rho-cells
        dy = (y_max - y_min) / ny_rho
        
        if grid_type in ['v', 'psi']:
            # Face/corner points - coordinates at cell faces
            # ny = ny_rho + 1 for faces
            y = y_min + np.arange(ny) * dy
        else:
            # Cell centers (rho, u, w)
            # ny = ny_rho for cell centers
            y = y_min + (np.arange(ny) + 0.5) * dy
        
        return y
    
    def _generate_z_coordinate(self, nz: int, z_type: str) -> np.ndarray:
        """
        Generate z-coordinate with appropriate staggering.
        
        Parameters
        ----------
        nz : int
            Number of points in z direction (already includes staggering from FAB header)
        z_type : str
            'rho' for cell centers or 'w' for cell faces
            
        Returns
        -------
        np.ndarray
            Z-coordinate array
        """
        z_min = self.domain_left[2] if len(self.domain_left) > 2 else 0.0
        z_max = self.domain_right[2] if len(self.domain_right) > 2 else 1.0
        
        if z_type == 'w':
            # Face points - nz points already from FAB header
            dz = (z_max - z_min) / (nz - 1) if nz > 1 else (z_max - z_min)
            z = z_min + np.arange(nz) * dz
        else:
            # Cell centers (rho-points)
            dz = (z_max - z_min) / nz
            z = z_min + (np.arange(nz) + 0.5) * dz
        
        return z
    
    def _get_coordinate_attrs(self, coord_name: str, axis: str, 
                             grid_type: str) -> Dict:
        """
        Get xgcm-compatible attributes for a coordinate.
        
        Parameters
        ----------
        coord_name : str
            Name like 'x_rho', 'z_w', etc.
        axis : str
            Axis identifier: 'X', 'Y', or 'Z'
        grid_type : str
            Grid type (rho, u, v, w, psi)
            
        Returns
        -------
        dict
            Coordinate attributes
        """
        attrs = {
            'long_name': f'{coord_name} coordinate',
            'axis': axis,
        }
        
        # Add c_grid_axis_shift for xgcm
        # Negative shift indicates the coordinate is at the "left" edge
        if grid_type == 'u' and axis == 'X':
            attrs['c_grid_axis_shift'] = -0.5
        elif grid_type == 'v' and axis == 'Y':
            attrs['c_grid_axis_shift'] = -0.5
        elif grid_type == 'w' and axis == 'Z':
            attrs['c_grid_axis_shift'] = -0.5
        elif grid_type == 'psi' and axis in ['X', 'Y']:
            attrs['c_grid_axis_shift'] = -0.5
        else:
            # Center points (rho)
            attrs['c_grid_axis_shift'] = 0.0
        
        return attrs
    
    def get_dimension_names(self, grid_type: str, is_2d: bool = False) -> Tuple[str, ...]:
        """
        Get dimension names for a variable on a specific grid.
        
        Parameters
        ----------
        grid_type : str
            Grid type (rho, u, v, w, psi)
        is_2d : bool
            Whether this is a 2D variable
            
        Returns
        -------
        tuple
            Dimension names like ('ocean_time', 'z_rho', 'y_rho', 'x_rho')
        """
        time_dim = 'ocean_time'
        
        if is_2d:
            # 2D variable: (time, y, x)
            return (time_dim, f'y_{grid_type}', f'x_{grid_type}')
        else:
            # 3D variable: (time, z, y, x)
            if grid_type == 'w':
                z_dim = 'z_w'
            else:
                z_dim = 'z_rho'
            
            return (time_dim, z_dim, f'y_{grid_type}', f'x_{grid_type}')
