"""
Automatic grid type detection from AMReX plotfile directory structure.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import re


class GridDetector:
    """
    Auto-detect grid types from plotfile directory structure.
    Scans *_H files to discover all available grids and variables.
    """
    
    # Map directory patterns to C-grid types
    GRID_TYPE_MAPPING = {
        'Cell': 'rho',      # Cell centers (rho-points)
        'UFace': 'u',       # X-face centers (u-points)
        'VFace': 'v',       # Y-face centers (v-points)
        'WFace': 'w',       # Z-face centers (w-points)
        'Nu_nd': 'psi',     # Node/corner points (psi-points)
        'rho2d': 'rho',     # 2D cell centers
        'u2d': 'rho',       # 2D barotropic u on rho-points (not staggered!)
        'v2d': 'rho',       # 2D barotropic v on rho-points (not staggered!)
        'psi2d': 'psi',     # 2D corners
    }
    
    def __init__(self):
        """Initialize grid detector."""
        pass
    
    def detect_grids(self, plotfile_path: Path, level: int = 0) -> Dict[str, Dict]:
        """
        Scan a level directory for all *_H files to discover grids.
        
        Parameters
        ----------
        plotfile_path : Path
            Path to the plotfile directory
        level : int, default 0
            AMR level to scan
            
        Returns
        -------
        dict
            Dictionary mapping directory name to grid information:
            {
                'Cell': {
                    'grid_type': 'rho',
                    'dimensionality': 3,
                    'num_components': 5,
                    'variables': ['temp', 'salt', ...],
                    'dimensions': (nx, ny, nz)
                },
                ...
            }
        """
        level_dir = plotfile_path / f"Level_{level}"
        
        if not level_dir.exists():
            raise ValueError(f"Level directory not found: {level_dir}")
        
        discovered_grids = {}
        
        # Find all *_H header files
        for header_file in sorted(level_dir.glob("*_H")):
            dir_name = header_file.stem  # e.g., "Cell_H", "UFace_H", "rho2d_H"
            # Remove the _H suffix to get the actual directory name
            if dir_name.endswith('_H'):
                dir_name = dir_name[:-2]
            
            try:
                grid_info = self._parse_grid_header(header_file)
                grid_info['grid_type'] = self._infer_grid_type(dir_name)
                grid_info['directory_name'] = dir_name
                discovered_grids[dir_name] = grid_info
            except Exception as e:
                print(f"Warning: Failed to parse {header_file}: {e}")
                continue
        
        return discovered_grids
    
    def _parse_grid_header(self, header_file: Path) -> Dict:
        """
        Parse a *_H header file to extract grid information.
        
        Parameters
        ----------
        header_file : Path
            Path to the header file (e.g., Cell_H, UFace_H)
            
        Returns
        -------
        dict
            Grid information including variables, dimensions, etc.
        """
        def read_nonempty_line(f):
            """Read next non-empty line, skipping blanks."""
            while True:
                line = f.readline()
                if not line:  # EOF
                    return None
                stripped = line.strip()
                if stripped:  # Non-empty
                    return stripped
        
        with open(header_file, 'r') as f:
            # Line 1: Version/format info (skip)
            read_nonempty_line(f)
            
            # Line 2: How data is written (skip)
            read_nonempty_line(f)
            
            # Line 3: Number of components (variables)
            num_components_line = read_nonempty_line(f)
            if num_components_line is None:
                raise ValueError(f"Could not find num_components in {header_file}")
            try:
                num_components = int(num_components_line)
            except ValueError as e:
                raise ValueError(f"Could not parse num_components from '{num_components_line}' in {header_file}: {e}")
            
            # Line 4: Number of ghost cells (skip for now)
            read_nonempty_line(f)
            
            # Line 5: Number of FABs
            nfabs_line = read_nonempty_line(f)
            if nfabs_line is None:
                raise ValueError(f"Could not find nfabs in {header_file}")
            
            # Format: (nfabs 0  or  ((nfabs) ...
            if nfabs_line.startswith('('):
                # Remove opening parenthesis and split
                parts = nfabs_line[1:].split()
                if parts:
                    try:
                        nfabs = int(parts[0])
                    except ValueError as e:
                        raise ValueError(f"Could not parse nfabs from '{parts[0]}' in {header_file}: {e}")
                else:
                    raise ValueError(f"Could not parse nfabs from: {nfabs_line}")
            else:
                raise ValueError(f"Unexpected format for nfabs line: {nfabs_line}")
            
            # Parse FAB index ranges to determine dimensions
            dimensions = self._parse_fab_dimensions(f, nfabs)
        
        # Determine dimensionality from dimensions
        dimensionality = sum(1 for d in dimensions if d > 1)
        
        return {
            'num_components': num_components,
            'num_fabs': nfabs,
            'dimensions': dimensions,
            'dimensionality': dimensionality,
            'variables': [],  # Will be filled from main Header
        }
    
    def _parse_fab_dimensions(self, f, nfabs: int) -> Tuple:
        """
        Parse FAB index ranges to determine overall grid dimensions.
        
        Parses the stagger tuple to get actual dimensions including staggering,
        and filters out zero-length dimensions for 2D variables.
        
        Parameters
        ----------
        f : file object
            Open file positioned at FAB index lines
        nfabs : int
            Number of FABs to parse
            
        Returns
        -------
        tuple
            Dimensions (nx, ny) for 2D or (nx, ny, nz) for 3D,
            with staggering already applied and zero-length dims removed
        """
        # Regex for parsing FAB indices
        _1dregx = r"-?\d+"
        _2dregx = r"-?\d+,-?\d+"
        _3dregx = r"-?\d+,-?\d+,-?\d+"
        
        # Parse format: ((lo_x,lo_y,lo_z) (hi_x,hi_y,hi_z) (stagger_x,stagger_y,stagger_z))
        dim_finder = re.compile(rf"\(\(({_3dregx})\) \(({_3dregx})\) \(({_3dregx})\)\)$")
        
        max_indices = [-1, -1, -1]
        stagger = [0, 0, 0]  # Will be same for all FABs in a grid
        
        for _ in range(nfabs):
            line = f.readline().strip()
            match = dim_finder.match(line)
            if match:
                lo_str, hi_str, stagger_str = match.groups()
                hi_indices = [int(x) for x in hi_str.split(',')]
                stagger_values = [int(x) for x in stagger_str.split(',')]
                
                for i in range(3):
                    max_indices[i] = max(max_indices[i], hi_indices[i])
                    stagger[i] = stagger_values[i]  # Should be same for all FABs
        
        # Convert from max index to dimension count
        # Dimensions already include staggering from the high index
        all_dims = [idx + 1 for idx in max_indices]
        
        # Filter out zero-length dimensions (for 2D variables)
        # If a dimension is 1 (from high index = 0), it's a zero-length dimension
        dimensions = tuple(d for d in all_dims if d > 1)
        
        return dimensions
    
    def _infer_grid_type(self, directory_name: str) -> str:
        """
        Infer C-grid type from directory name.
        
        Parameters
        ----------
        directory_name : str
            Name like 'Cell', 'UFace', 'rho2d', etc.
            
        Returns
        -------
        str
            Grid type: 'rho', 'u', 'v', 'w', or 'psi'
        """
        # Check against known patterns
        for pattern, grid_type in self.GRID_TYPE_MAPPING.items():
            if pattern in directory_name:
                return grid_type
        
        # Default to rho-points if unknown
        return 'rho'
    
    def get_all_variables_from_main_header(self, plotfile_path: Path, 
                                          grid_mapping: Dict[str, List[str]] = None) -> Dict[str, str]:
        """
        Read variable names from main Header file and map to grid directories.
        
        Parameters
        ----------
        plotfile_path : Path
            Path to plotfile directory
        grid_mapping : dict, optional
            Manual mapping of variable names to grid directories
            If None, attempts automatic detection
            
        Returns
        -------
        dict
            Mapping of variable name to directory name
        """
        header_file = plotfile_path / "Header"
        
        with open(header_file, 'r') as f:
            # Skip version line if present
            first_line = f.readline().strip()
            if any(c.isalpha() for c in first_line):
                n_fields = int(f.readline().strip())
            else:
                n_fields = int(first_line)
            
            # Read variable names
            variables = [f.readline().strip() for _ in range(n_fields)]
        
        return variables
