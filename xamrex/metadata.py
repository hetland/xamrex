"""
Enhanced metadata parsing for multi-grid AMReX plotfiles.
Tracks level availability across time series and discovers all grids.
"""
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np

from .grid_detector import GridDetector


class AMReXBasicMeta:
    """
    Basic metadata parser for a single AMReX plotfile.
    Reads the main Header file for dataset-wide information.
    """
    
    def __init__(self, plotfile_path: Path):
        """
        Parse basic metadata from plotfile Header.
        
        Parameters
        ----------
        plotfile_path : Path
            Path to plotfile directory
        """
        self.plotfile_path = Path(plotfile_path)
        self.header_file = self.plotfile_path / "Header"
        
        if not self.header_file.exists():
            raise FileNotFoundError(f"Header file not found: {self.header_file}")
        
        self._parse_header()
    
    def _parse_header(self):
        """Parse the main Header file."""
        with open(self.header_file, 'r') as f:
            # Line 1: Version or number of fields
            first_line = f.readline().strip()
            if any(c.isalpha() for c in first_line):
                self.version = first_line
                self.n_fields = int(f.readline().strip())
            else:
                self.version = None
                self.n_fields = int(first_line)
            
            # Read field names from main section
            self.field_list = [f.readline().strip() for _ in range(self.n_fields)]
            
            # Dimensionality
            self.dimensionality = int(f.readline().strip())
            
            # Current time
            self.current_time = float(f.readline().strip())
            
            # Max level
            self.max_level = int(f.readline().strip())
            
            # Domain edges
            domain_left = np.array(f.readline().strip().split(), dtype=float)
            self.domain_left_edge = domain_left
            
            domain_right = np.array(f.readline().strip().split(), dtype=float)
            self.domain_right_edge = domain_right
            
            # Refinement factors
            ref_line = f.readline().strip()
            if ref_line:
                self.ref_factors = np.array(ref_line.split(), dtype=int)
            else:
                self.ref_factors = np.array([2] * (self.max_level + 1), dtype=int)
            
            # Level dimensions from index space
            index_space_line = f.readline().strip()
            self.level_dimensions = self._parse_level_dimensions(index_space_line)
            
            # Skip time step and cycle info
            f.readline()  # time step
            f.readline()  # cycle or blank
            
            # Skip coordinate system lines
            f.readline()  # coordinate 0 line
            f.readline()  # coordinate 1 line
            
            # Skip domain bounds
            for _ in range(self.dimensionality):
                f.readline()
            
            # Now parse grid-specific variable listings
            self.variable_to_grid_map = self._parse_variable_grid_mapping(f)
    
    def _parse_level_dimensions(self, index_space_line: str) -> Dict[int, List[int]]:
        """
        Parse level dimensions from index space line.
        
        Example: "((0,0,0) (37,47,31) (0,0,0)) ((0,0,0) (113,143,31) (0,0,0))"
        
        Returns
        -------
        dict
            {level: [nx, ny, nz]} for ALL levels present in the plotfile
        """
        import re
        
        # Extract middle coordinates for each level
        pattern = r'\(\([^)]+\) \(([^)]+)\) \([^)]+\)\)'
        matches = re.findall(pattern, index_space_line)
        
        level_dims = {}
        for level, match in enumerate(matches):
            coords = [int(c.strip()) for c in match.split(',')]
            # Convert from max index to dimension count
            dims = [c + 1 for c in coords[:self.dimensionality]]
            level_dims[level] = dims
        
        return level_dims
    
    def _parse_variable_grid_mapping(self, f) -> Dict[str, str]:
        """
        Parse the grid-specific variable listings from the Header file.
        
        State machine approach to handle both single-level and multi-level formats.
        
        Single-level format:
            Level_0/Cell
            4
            3
            amrexvec_nu_x
            ...
            Level_0/Nu_nd
            
        Multi-level format:
            Level_0/Cell
            1 1 4320
            382
            <numeric data>
            Level_1/Cell
            4
            3
            amrexvec_nu_x
            ...
            Level_0/Nu_nd
            Level_1/Nu_nd
        
        Parameters
        ----------
        f : file object
            Open file positioned after domain bounds
            
        Returns
        -------
        dict
            {variable_name: grid_directory_name}
        """
        var_to_grid = {}
        all_variables = list(self.field_list)
        
        # Main field_list variables are in Cell by default
        for var in self.field_list:
            var_to_grid[var] = 'Cell'
        
        # Find start of variable section (skip Level_0/Cell metadata)
        while True:
            line = f.readline()
            if not line:
                self.field_list = all_variables
                self.variable_to_component_index = self._create_component_index_mapping(var_to_grid)
                return var_to_grid
            
            stripped = line.strip()
            # Look for Level_X/Cell where X might be > 0 for multi-level
            if stripped.startswith('Level_') and '/Cell' in stripped:
                # Check if next line is a single digit (variable count)
                next_line = f.readline()
                if not next_line:
                    break
                next_stripped = next_line.strip()
                
                # Single digit = start of variables, multi-part number = metadata
                if next_stripped and len(next_stripped.split()) == 1 and next_stripped.isdigit():
                    # Found variable section! Start parsing
                    count = int(next_stripped)
                    vars_for_cell, grid_name = self._parse_variable_block(f, count)
                    for var in vars_for_cell:
                        var_to_grid[var] = grid_name
                        if var not in all_variables:
                            all_variables.append(var)
                    break
        
        # Continue parsing remaining grids
        while True:
            # Read next count
            count_line = f.readline()
            if not count_line:
                break
            count_line = count_line.strip()
            
            # Skip empty lines and metadata
            if not count_line:
                continue
            
            # Skip Level_ lines (we'll hit them after reading variables)
            if count_line.startswith('Level_'):
                continue
            
            # Skip multi-part numeric lines (metadata)
            parts = count_line.split()
            if len(parts) > 1:
                continue
            
            # Check if it's a valid count (single positive integer)
            if not parts[0].isdigit():
                continue
            
            count = int(parts[0])
            if count <= 0 or count > 100:  # Sanity check
                continue
            
            # Parse variable block
            vars_for_grid, grid_name = self._parse_variable_block(f, count)
            
            if grid_name:
                for var in vars_for_grid:
                    var_to_grid[var] = grid_name
                    if var not in all_variables:
                        all_variables.append(var)
        
        # Update field_list to include all variables
        self.field_list = all_variables
        self.variable_to_component_index = self._create_component_index_mapping(var_to_grid)
        return var_to_grid
    
    def _parse_variable_block(self, f, count: int) -> Tuple[List[str], str]:
        """
        Parse a block of variables and their grid label.
        
        Parameters
        ----------
        f : file object
            Open file positioned after count line
        count : int
            Number of variables to read
            
        Returns
        -------
        tuple
            (list of variable names, grid name)
        """
        variables = []
        
        # Read variable names
        for _ in range(count):
            var_line = f.readline()
            if not var_line:
                break
            var_line = var_line.strip()
            
            # Skip Level_ lines and numeric lines
            if var_line and not var_line.startswith('Level_'):
                # Check if it's not a numeric line
                try:
                    float(var_line.split()[0])
                    continue  # Skip numeric lines
                except (ValueError, IndexError):
                    variables.append(var_line)
        
        # Read Level_X/GridName lines
        # In single-level format: 1 line (Level_0/GridName)
        # In multi-level format: max_level+1 lines (Level_0/GridName, Level_1/GridName, ...)
        grid_name = None
        
        # Read exactly max_level + 1 lines (or until we hit a non-Level line)
        for _ in range(self.max_level + 1):
            label_line = f.readline()
            if not label_line:
                break
            label_line = label_line.strip()
            
            if label_line.startswith('Level_') and '/' in label_line:
                if grid_name is None:
                    # Extract grid name from first Level_X line
                    grid_name = label_line.split('/')[1]
                # Continue to consume all Level_ lines
            else:
                # Hit non-Level line - this should not happen in well-formed files
                # but we'll break anyway
                break
        
        return variables, grid_name if grid_name else ''
    
    def _create_component_index_mapping(self, var_to_grid: Dict[str, str]) -> Dict[str, int]:
        """
        Create mapping from variable name to its component index within its grid.
        
        Variables are listed in order in the Header, so we track the order
        within each grid directory.
        
        Returns
        -------
        dict
            {variable_name: component_index}
        """
        # Track component counter for each grid
        grid_component_counter = {}
        var_to_comp_idx = {}
        
        # Process variables in the order they appear in field_list
        for var_name in self.field_list:
            grid = var_to_grid.get(var_name, 'Cell')
            
            # Initialize counter for this grid if needed
            if grid not in grid_component_counter:
                grid_component_counter[grid] = 0
            
            # Assign component index
            var_to_comp_idx[var_name] = grid_component_counter[grid]
            
            # Increment counter for this grid
            grid_component_counter[grid] += 1
        
        return var_to_comp_idx


class AMReXMultiGridMeta:
    """
    Multi-grid metadata parser for AMReX plotfiles.
    
    Discovers all grids, tracks level availability across time series,
    and manages variable-to-grid mappings.
    """
    
    def __init__(self, plotfile_paths: List[Path]):
        """
        Initialize multi-grid metadata parser.
        
        Parameters
        ----------
        plotfile_paths : list of Path
            All plotfiles in the time series
        """
        self.plotfile_paths = [Path(p) for p in plotfile_paths]
        self.detector = GridDetector()
        
        # Parse first file for basic info
        self.basic_meta = AMReXBasicMeta(self.plotfile_paths[0])
        
        # Survey all files
        self.level_availability = self._survey_level_availability()
        self.time_values = self._get_all_times()
        
        # Merge level_dimensions from all files to get superset
        self._merge_level_dimensions()
        
        # Discover all grids across all files and levels
        self.all_grids = self._discover_all_grids()
        
        # Map variables to grids
        self.variable_to_grid = self._map_variables_to_grids()
        
        # Determine max level that ever appears
        self.max_level_ever = max(self.level_availability.keys())
    
    def _survey_level_availability(self) -> Dict[int, List[int]]:
        """
        Survey which levels exist in which files.
        
        Returns
        -------
        dict
            {level: [file_indices where this level exists]}
        """
        level_map = {}
        
        for file_idx, pf_path in enumerate(self.plotfile_paths):
            try:
                meta = AMReXBasicMeta(pf_path)
                for level in range(meta.max_level + 1):
                    # Check if level directory actually exists
                    level_dir = pf_path / f"Level_{level}"
                    if level_dir.exists():
                        if level not in level_map:
                            level_map[level] = []
                        level_map[level].append(file_idx)
            except Exception as e:
                print(f"Warning: Failed to survey {pf_path}: {e}")
                continue
        
        return level_map
    
    def _get_all_times(self) -> np.ndarray:
        """Get time values from all plotfiles."""
        times = []
        for pf_path in self.plotfile_paths:
            try:
                meta = AMReXBasicMeta(pf_path)
                times.append(meta.current_time)
            except Exception:
                times.append(np.nan)
        
        return np.array(times)
    
    def _merge_level_dimensions(self):
        """
        Merge level_dimensions from all plotfiles to get the superset.
        
        This ensures that if different files have different max levels,
        we get dimensions for all levels that ever appear.
        """
        merged_dims = {}
        
        for pf_path in self.plotfile_paths:
            try:
                meta = AMReXBasicMeta(pf_path)
                # Merge this file's level_dimensions into the superset
                for level, dims in meta.level_dimensions.items():
                    if level not in merged_dims:
                        merged_dims[level] = dims
                    # If level already exists, verify dimensions match
                    elif merged_dims[level] != dims:
                        print(f"Warning: Level {level} dimensions differ across files: "
                              f"{merged_dims[level]} vs {dims}. Using first encountered.")
            except Exception as e:
                print(f"Warning: Failed to parse level dimensions from {pf_path}: {e}")
                continue
        
        # Update basic_meta with merged dimensions
        self.basic_meta.level_dimensions = merged_dims
    
    def _discover_all_grids(self) -> Dict[str, Dict]:
        """
        Discover all unique grids across all files and levels.
        
        Returns
        -------
        dict
            {directory_name: grid_info_dict}
        """
        all_grids = {}
        
        # Survey each level that appears anywhere
        for level in self.level_availability.keys():
            # Use first file where this level appears
            file_idx = self.level_availability[level][0]
            pf_path = self.plotfile_paths[file_idx]
            
            try:
                grids = self.detector.detect_grids(pf_path, level)
                
                # Merge with existing (union of all grids)
                for dir_name, grid_info in grids.items():
                    if dir_name not in all_grids:
                        all_grids[dir_name] = grid_info.copy()
                        all_grids[dir_name]['levels_present'] = {level}
                    else:
                        # Track which levels have this grid
                        all_grids[dir_name]['levels_present'].add(level)
                        
            except Exception as e:
                print(f"Warning: Failed to discover grids in {pf_path} level {level}: {e}")
                continue
        
        return all_grids
    
    def _map_variables_to_grids(self) -> Dict[str, str]:
        """
        Map variable names to grid directory names.
        
        Uses the variable_to_grid_map parsed from the Header file,
        with fallback to Cell for variables only in main field list.
        
        Returns
        -------
        dict
            {variable_name: directory_name}
        """
        var_to_grid = {}
        
        # Use the mapping from the Header file
        if hasattr(self.basic_meta, 'variable_to_grid_map'):
            var_to_grid.update(self.basic_meta.variable_to_grid_map)
        
        # For any variables in field_list not yet mapped, default to Cell
        for var_name in self.basic_meta.field_list:
            if var_name not in var_to_grid:
                var_to_grid[var_name] = 'Cell'
        
        return var_to_grid
    
    def get_grid_info(self, directory_name: str) -> Dict:
        """Get grid information for a specific directory."""
        return self.all_grids.get(directory_name, {})
    
    def get_variable_grid(self, variable_name: str) -> str:
        """Get the grid directory name for a variable."""
        return self.variable_to_grid.get(variable_name, 'Cell')
    
    def is_level_available(self, level: int, time_index: int) -> bool:
        """Check if a level is available at a specific time index."""
        if level not in self.level_availability:
            return False
        return time_index in self.level_availability[level]
