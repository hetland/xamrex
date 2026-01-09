"""
FAB (File Array Block) data loading with masking support.
Adapted from xamrex with multi-grid and masking enhancements.
"""
from pathlib import Path
from typing import Dict, Tuple
import struct
import numpy as np
import pandas as pd
import dask.array as da
from dask import delayed
import re

class FABMetadata:
    """
    Metadata for FABs in a specific grid directory at a specific level.
    """
    
    def __init__(self, plotfile_path: Path, level: int, directory_name: str, 
                 num_components: int, dimensionality: int):
        """
        Parse FAB metadata from *_H file.
        
        Parameters
        ----------
        plotfile_path : Path
            Path to plotfile directory
        level : int
            AMR level
        directory_name : str
            Directory name (e.g., 'Cell', 'UFace')
        num_components : int
            Number of variables/components
        dimensionality : int
            2 or 3
        """
        self.plotfile_path = Path(plotfile_path)
        self.level = level
        self.directory_name = directory_name
        self.num_components = num_components
        self.dimensionality = dimensionality
        
        self.header_file = self.plotfile_path / f"Level_{level}" / f"{directory_name}_H"
        
        if not self.header_file.exists():
            raise FileNotFoundError(f"Header file not found: {self.header_file}")
        
        self._parse_header()
    
    def _parse_header(self):
        """Parse the *_H header file."""
        with open(self.header_file, 'r') as f:
            # Skip first two lines
            f.readline()
            f.readline()
            
            # Verify num components
            n_comp = int(f.readline().strip())
            if n_comp != self.num_components:
                print(f"Warning: Component count mismatch: {n_comp} vs {self.num_components}")
            
            # Number of ghost cells - can be a scalar or tuple
            ghost_line = f.readline().strip()
            if ghost_line.startswith('('):
                # Tuple format like "(1,1,0)" for 2D grids
                # Extract the values and use the first one
                ghost_vals = ghost_line.strip('()').split(',')
                self.nghost = int(ghost_vals[0])
            else:
                # Scalar format like "0" for 3D grids
                self.nghost = int(ghost_line)
            
            # Number of FABs
            nfabs_line = f.readline().strip()
            # Format: "(nfabs 0" or "((nfabs) ..."
            parts = nfabs_line.split()
            if parts[0].startswith('('):
                # Remove leading parentheses
                nfabs_str = parts[0].lstrip('(')
                self.nfabs = int(nfabs_str)
            else:
                raise ValueError(f"Unexpected nfabs line format: {nfabs_line}")
            
            # Parse FAB indices
            # Note: FAB headers ALWAYS use 3D format ((x,y,z) (x,y,z) (x,y,z))
            # even for 2D grids (where z coordinates are 0)
            _3dregx = r"-?\d+,-?\d+,-?\d+"
            dim_finder = re.compile(rf"\(\(({_3dregx})\) \(({_3dregx})\) \({_3dregx}\)\)$")
            
            fab_inds_lo = np.zeros((self.nfabs, 3), dtype=int)
            fab_inds_hi = np.zeros((self.nfabs, 3), dtype=int)
            
            for fabnum in range(self.nfabs):
                line = f.readline().strip()
                match = dim_finder.match(line)
                if match:
                    start, stop = match.groups()
                    start = np.array(start.split(","), dtype=int)
                    stop = np.array(stop.split(","), dtype=int)
                    fab_inds_lo[fabnum, :] = start
                    fab_inds_hi[fabnum, :] = stop + 1  # Python-style indexing
            
            # Skip closing parenthesis and blank line
            f.readline()
            f.readline()
            
            # Read filenames and byte offsets
            fab_filename = np.zeros(self.nfabs, dtype=object)
            fab_byte_offset = np.zeros(self.nfabs, dtype=int)
            
            for fabnum in range(self.nfabs):
                # Read the FabOnDisk line
                line = f.readline().strip()
                
                # Parse format: "FabOnDisk: filename offset" or "1\nFabOnDisk: filename offset"
                if not line.startswith('FabOnDisk:'):
                    # May need to read another line
                    line = f.readline().strip()
                
                if line.startswith('FabOnDisk:'):
                    parts = line.split()
                    fab_filename[fabnum] = parts[1]
                    fab_byte_offset[fabnum] = int(parts[2])
                else:
                    raise ValueError(f"Expected 'FabOnDisk:' line, got: {line}")
                
                # Skip additional metadata lines (blank, dim info, values, blank)
                # These appear to be min/max or other statistics
                f.readline()  # blank line
                dim_line = f.readline().strip()  # e.g., "1,5"
                if dim_line:  # If there's dimension info
                    f.readline()  # value line
                    f.readline()  # blank line
        
        # Build metadata DataFrame
        df_cols = ['lo_i', 'lo_j', 'lo_k', 'hi_i', 'hi_j', 'hi_k', 
                   'filename', 'byte_offset']
        metadata = pd.DataFrame(columns=df_cols)
        metadata.index.name = 'fab_id'
        
        metadata['lo_i'] = fab_inds_lo[:, 0]
        metadata['lo_j'] = fab_inds_lo[:, 1]
        metadata['lo_k'] = fab_inds_lo[:, 2]
        metadata['hi_i'] = fab_inds_hi[:, 0]
        metadata['hi_j'] = fab_inds_hi[:, 1]
        metadata['hi_k'] = fab_inds_hi[:, 2]
        
        metadata['filename'] = fab_filename
        metadata['byte_offset'] = fab_byte_offset
        
        # Calculate fab dimensions
        metadata['di'] = metadata['hi_i'] - metadata['lo_i']
        metadata['dj'] = metadata['hi_j'] - metadata['lo_j']
        metadata['dk'] = metadata['hi_k'] - metadata['lo_k']
        metadata['ncells'] = metadata['di'] * metadata['dj'] * metadata['dk']
        
        self.metadata = metadata

class FABLoader:
    """
    Loads data from FABs with lazy evaluation and masking support.
    """
    
    def __init__(self, plotfile_path: Path, level: int, directory_name: str,
                 variable_index: int, fab_metadata: FABMetadata, 
                 full_shape: Tuple[int, ...]):
        """
        Initialize FAB loader for a specific variable.
        
        Parameters
        ----------
        plotfile_path : Path
            Path to plotfile directory
        level : int
            AMR level
        directory_name : str
            Directory name (e.g., 'Cell', 'UFace')
        variable_index : int
            Index of variable in the FAB file
        fab_metadata : FABMetadata
            FAB metadata object
        full_shape : tuple
            Full array shape including time dimension
        """
        self.plotfile_path = Path(plotfile_path)
        self.level = level
        self.directory_name = directory_name
        self.variable_index = variable_index
        self.fab_metadata = fab_metadata
        self.full_shape = full_shape
        self.dtype = np.float64
    
    def create_dask_array(self) -> da.Array:
        """Create a dask array that lazily loads the data."""
        @delayed
        def load_data():
            return self._load_full_field_data()
        
        lazy_data = load_data()
        return da.from_delayed(lazy_data, shape=self.full_shape, dtype=self.dtype)
    
    def _load_full_field_data(self) -> np.ndarray:
        """Load complete field data from all FABs."""
        # Initialize with NaN for missing values
        full_data = np.full(self.full_shape, np.nan, dtype=self.dtype)
        
        # Load data from each FAB
        for fab_idx in range(self.fab_metadata.nfabs):
            fab_data = self._read_fab_data(fab_idx)
            
            # Get index ranges
            fab_row = self.fab_metadata.metadata.iloc[fab_idx]
            lo_i, lo_j, lo_k = fab_row['lo_i'], fab_row['lo_j'], fab_row['lo_k']
            hi_i, hi_j, hi_k = fab_row['hi_i'], fab_row['hi_j'], fab_row['hi_k']
            
            # Place data in correct location
            if self.fab_metadata.dimensionality == 3:
                # 3D: (time, z, y, x)
                full_data[0, lo_k:hi_k, lo_j:hi_j, lo_i:hi_i] = fab_data
            else:
                # 2D: (time, y, x)
                full_data[0, lo_j:hi_j, lo_i:hi_i] = fab_data
        
        return full_data
    
    def _read_fab_data(self, fab_idx: int) -> np.ndarray:
        """Read data for a specific FAB."""
        fab_row = self.fab_metadata.metadata.iloc[fab_idx]
        fab_file = (self.plotfile_path / f"Level_{self.level}" / 
                   self.directory_name / fab_row['filename'])
        
        # If directory structure is different (data files at level root)
        if not fab_file.exists():
            fab_file = (self.plotfile_path / f"Level_{self.level}" / 
                       fab_row['filename'])
        
        di, dj, dk = fab_row['di'], fab_row['dj'], fab_row['dk']
        cells_per_fab = di * dj * dk
        
        with open(fab_file, 'rb') as f:
            # Skip to byte offset
            f.seek(fab_row['byte_offset'])
            
            # Skip metadata line
            f.readline()
            data_start = f.tell()
            
            # Calculate field start position
            field_start = data_start + self.variable_index * cells_per_fab * 8
            f.seek(field_start)
            
            # Read field data
            field_bytes = f.read(cells_per_fab * 8)
            
            if len(field_bytes) == cells_per_fab * 8:
                field_data = struct.unpack(f'<{cells_per_fab}d', field_bytes)
            else:
                # Fallback
                field_data = []
                f.seek(field_start)
                for _ in range(cells_per_fab):
                    value_bytes = f.read(8)
                    if len(value_bytes) == 8:
                        value = struct.unpack('<d', value_bytes)[0]
                        field_data.append(value)
                    else:
                        field_data.append(0.0)
            
            # Reshape
            if self.fab_metadata.dimensionality == 3:
                fab_data = np.array(field_data, dtype=np.float64).reshape((dk, dj, di))
            else:
                fab_data = np.array(field_data, dtype=np.float64).reshape((dj, di))
        
        return fab_data

class MaskedFABLoader:
    """
    Creates masked arrays when a level doesn't exist at a timestep.
    """
    
    def __init__(self, full_shape: Tuple[int, ...]):
        """
        Initialize masked FAB loader.
        
        Parameters
        ----------
        full_shape : tuple
            Full array shape to create
        """
        self.full_shape = full_shape
        self.dtype = np.float64
    
    def create_dask_array(self) -> da.Array:
        """Create a dask array filled with NaN."""
        @delayed
        def create_masked():
            return np.full(self.full_shape, np.nan, dtype=self.dtype)
        
        lazy_data = create_masked()
        return da.from_delayed(lazy_data, shape=self.full_shape, dtype=self.dtype)
