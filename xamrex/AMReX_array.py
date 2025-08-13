"""
Core AMReX plotfile parsing utilities.
Contains metadata parsers for AMReX plotfiles without the legacy dask array implementation.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import re


class AMReXDatasetMeta():
    """
    High-level metadata for an entire AMReX plotfile.
    
    Parses the main Header file to extract dataset-wide information.
    Based on the BoxlibDataset class from yt.
    """
    
    def __init__(self, fplt, time_dimension_name=None, dimension_names=None):
        """
        Initialize dataset metadata parser.
        
        Parameters
        ----------
        fplt : str or Path
            Path to the plotfile directory
        time_dimension_name : str, optional
            Name for the time dimension (default: 'ocean_time')
        dimension_names : dict, optional
            Custom dimension names, e.g., {'x': 'longitude', 'y': 'latitude', 'z': 'depth'}
        """
        self.fplt = Path(fplt)
        self.fdataset_header = Path(fplt, 'Header')
        
        # FIXED ISSUE 1: Allow configurable dimension names
        self.time_dimension_name = time_dimension_name or 'ocean_time'
        self.dimension_names = dimension_names or {}
        
        # Domain offset; doesn't necessarily need to equal (0,0,0)
        self.domain_offset = np.zeros(3, dtype="int64")
        self.geometry = "cartesian"  # TODO: could be extracted from header if needed
        
        self._parse_dataset_header()
        
    def _parse_dataset_header(self):
        """Parse the Header file to extract dataset metadata."""
        with open(self.fdataset_header) as header_file:
            # Skip version identifier if present (e.g., "HyperCLaw-V1.1")
            first_line = header_file.readline().strip()
            
            # Check if first line is a version identifier (contains letters)
            if any(c.isalpha() for c in first_line):
                # First line is version, read number of fields from next line
                self.version = first_line
                self.n_fields = int(header_file.readline())
            else:
                # First line is number of fields (old format)
                self.version = None
                self.n_fields = int(first_line)
            
            # Parse the header file line by line
            self.field_list = [header_file.readline().strip() for i in range(self.n_fields)]
            self.dimensionality = int(header_file.readline())
            self.current_time = float(header_file.readline())
            self.max_level = int(header_file.readline())
            
            # Domain edges
            domain_left_edge = np.zeros(self.dimensionality, dtype=float)
            domain_left_edge[:] = header_file.readline().split()
            self.domain_left_edge = domain_left_edge
            
            domain_right_edge = np.zeros(self.dimensionality, dtype=float)
            domain_right_edge[:] = header_file.readline().split()
            self.domain_right_edge = domain_right_edge

            # Refinement factors (typically 2 for AMR-Wind)
            ref_factors = np.array(header_file.readline().split(), dtype="int64")
            if ref_factors.size == 0:
                # Use default of 2
                ref_factors = [2] * (self.max_level + 1)
            
            self.ref_factors = ref_factors
            
            # Handle varying refinement factors (edge case)
            if np.unique(ref_factors).size > 1:
                # We want everything to be a multiple of this
                self.refine_by = min(ref_factors)
                # Check that they're all multiples of the minimum
                if not all(
                    float(rf) / self.refine_by == int(float(rf) / self.refine_by)
                    for rf in ref_factors
                ):
                    raise RuntimeError("Refinement factors must be multiples of the minimum")
                base_log = np.log2(self.refine_by)
                self.level_offsets = [0]  # level 0 has to have 0 offset
                lo = 0
                for rf in self.ref_factors:
                    lo += int(np.log2(rf) / base_log) - 1
                    self.level_offsets.append(lo)
            else:
                self.refine_by = ref_factors[0]
                self.level_offsets = [0 for l in range(self.max_level + 1)]
            
            # Global index space
            index_space = header_file.readline()
            # Format: ((0,0,0) (255,255,255) (0,0,0)) ((0,0,0) (511,511,511) (0,0,0))
            root_space = index_space.replace("(", "").replace(")", "").split()[:2]
            start = np.array(root_space[0].split(","), dtype="int64")
            stop = np.array(root_space[1].split(","), dtype="int64")
            dd = np.ones(3, dtype="int64")
            dd[:self.dimensionality] = stop - start + 1
            self.domain_offset[:self.dimensionality] = start  # Level 0 offset
            self.domain_dimensions = dd  # Level 0 dimensions


class AMReXFabsMetaSingleLevel():
    """
    Metadata parser for fabs (file array blocks) at a single AMR level.
    
    Collects fab index ranges, filenames, and byte offsets for data loading.
    """
    
    def __init__(self, fplt, n_fields, dimensionality, level):
        """
        Initialize fab metadata parser for a specific level.
        
        Parameters
        ----------
        fplt : str or Path
            Path to the plotfile directory
        n_fields : int
            Number of fields in the dataset
        dimensionality : int
            Spatial dimensionality (2 or 3)
        level : int
            AMR level to parse
        """
        self.fplt = Path(fplt)
        self.n_fields = n_fields
        self.dimensionality = dimensionality
        self.level = level
        self.fheader_file = Path(self.fplt, f'Level_{level}/Cell_H')
        
        self._parse_level_header()
        
    def _parse_level_header(self):
        """Parse the Cell_H file for this refinement level."""
        with open(self.fheader_file) as header_file:
            # Skip first two lines (header file version, how data was written)
            header_file.readline()
            header_file.readline()
            
            # Verify number of fields matches
            cell_h_n_fields = int(header_file.readline())
            if self.n_fields != cell_h_n_fields:
                raise ValueError(f"Field count mismatch: {self.n_fields} vs {cell_h_n_fields}")
            
            # Number of ghost cells
            self.nghost = int(header_file.readline())
            
            # Number of FABs in this level
            self.nfabs = int(header_file.readline().split()[0][1:])
            
            # Set up regex patterns for parsing fab indices
            _1dregx = r"-?\d+"
            _2dregx = r"-?\d+,-?\d+"
            _3dregx = r"-?\d+,-?\d+,-?\d+"
            _dim_finder = [
                re.compile(rf"\(\(({ndregx})\) \(({ndregx})\) \({ndregx}\)\)$")
                for ndregx in (_1dregx, _2dregx, _3dregx)
            ]
            _our_dim_finder = _dim_finder[self.dimensionality - 1]

            # Collect fab index ranges
            fab_inds_lo = np.zeros((self.nfabs, self.dimensionality), dtype=int)
            fab_inds_hi = np.zeros((self.nfabs, self.dimensionality), dtype=int)
            for fabnum in range(self.nfabs):
                start, stop = _our_dim_finder.match(header_file.readline()).groups()
                start = np.array(start.split(","), dtype=int)
                stop = np.array(stop.split(","), dtype=int)
                fab_inds_lo[fabnum, :] = start
                fab_inds_hi[fabnum, :] = stop + 1  # Python-style indexing

            # Verify we read all fab indices
            endcheck = header_file.readline()
            if endcheck != ')\n':
                raise ValueError(f"Failed to read all fab indices. Next line: '{endcheck}'")
            header_file.readline()  # Skip next line
            
            # Collect filenames and byte offsets
            fab_filename = np.zeros(self.nfabs, dtype=object)
            fab_byte_offset = np.zeros(self.nfabs, dtype=int)
            for fabnum in range(self.nfabs):
                _, filename, byte_offset = header_file.readline().split()
                fab_filename[fabnum] = filename
                fab_byte_offset[fabnum] = int(byte_offset)
                
            # Store fab info in a DataFrame for easy manipulation
            df_cols = ['lo_i', 'lo_j', 'lo_k', 'hi_i', 'hi_j', 'hi_k', 'filename', 'byte_offset']
            df_meta = pd.DataFrame(columns=df_cols)
            df_meta.index.name = 'fab_id'
            
            # Fill in the fab data
            df_meta['lo_i'] = fab_inds_lo[:, 0]
            df_meta['lo_j'] = fab_inds_lo[:, 1] 
            df_meta['lo_k'] = fab_inds_lo[:, 2]
            df_meta['hi_i'] = fab_inds_hi[:, 0]
            df_meta['hi_j'] = fab_inds_hi[:, 1]
            df_meta['hi_k'] = fab_inds_hi[:, 2]
            df_meta['filename'] = fab_filename
            df_meta['byte_offset'] = fab_byte_offset
            
            # Calculate derived fab properties
            df_meta['di'] = df_meta['hi_i'] - df_meta['lo_i']
            df_meta['dj'] = df_meta['hi_j'] - df_meta['lo_j']
            df_meta['dk'] = df_meta['hi_k'] - df_meta['lo_k']
            df_meta['ncells'] = df_meta['di'] * df_meta['dj'] * df_meta['dk']
                
            # Initialize data_offset column - will be computed lazily when needed
            df_meta['data_offset'] = None
                
            # Store the completed metadata
            self.metadata = df_meta
    
    def get_fab_data_offset(self, fab_index):
        """
        Get the data offset for a specific fab, computing it lazily if needed.
        
        Parameters
        ----------
        fab_index : int
            Index of the fab in the metadata DataFrame
            
        Returns
        -------
        int
            Byte offset to the start of data in the fab file
        """
        if self.metadata.loc[fab_index, 'data_offset'] is None:
            # Compute data offset by reading the fab file
            fab_row = self.metadata.loc[fab_index]
            fab_file = self.fplt / f"Level_{self.level}" / fab_row['filename']
            
            with open(fab_file, 'rb') as f:
                f.seek(fab_row['byte_offset'])
                f.readline()  # Skip the metadata line
                data_offset = f.tell()
            
            # Cache the result
            self.metadata.loc[fab_index, 'data_offset'] = data_offset
        
        return self.metadata.loc[fab_index, 'data_offset']
