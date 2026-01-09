"""
Unit tests for xamrex package components.
"""
import pytest
from pathlib import Path
import numpy as np
import xarray as xr

# Test data location
TEST_DATA = Path(__file__).parent.parent / "ocean_out"
PLOTFILE_0 = TEST_DATA / "plt00000"
PLOTFILE_360 = TEST_DATA / "plt00360"
ALL_PLOTFILES = sorted(TEST_DATA.glob("plt*"))

# Skip tests if test data not available
pytestmark = pytest.mark.skipif(
    not TEST_DATA.exists(),
    reason="Test data (ocean_out) not available"
)


class TestGridDetector:
    """Test automatic grid type detection."""
    
    def test_detect_grids_level_0(self):
        """Test grid detection at level 0."""
        from xamrex.grid_detector import GridDetector
        
        detector = GridDetector()
        grids = detector.detect_grids(PLOTFILE_0, level=0)
        
        # Should find multiple grid types
        assert len(grids) > 0
        assert 'Cell' in grids or 'UFace' in grids
        
        # Check grid info structure
        for dir_name, grid_info in grids.items():
            assert 'grid_type' in grid_info
            assert 'dimensionality' in grid_info
            assert 'num_components' in grid_info
            assert grid_info['grid_type'] in ['rho', 'u', 'v', 'w', 'psi']
    
    def test_grid_type_mapping(self):
        """Test grid type inference from directory names."""
        from xamrex.grid_detector import GridDetector
        
        detector = GridDetector()
        
        assert detector._infer_grid_type('Cell') == 'rho'
        assert detector._infer_grid_type('UFace') == 'u'
        assert detector._infer_grid_type('VFace') == 'v'
        assert detector._infer_grid_type('WFace') == 'w'
        assert detector._infer_grid_type('Nu_nd') == 'psi'
        assert detector._infer_grid_type('rho2d') == 'rho'
        assert detector._infer_grid_type('u2d') == 'u'


class TestMetadata:
    """Test metadata parsing."""
    
    def test_basic_meta_single_file(self):
        """Test basic metadata from a single file."""
        from xamrex.metadata import AMReXBasicMeta
        
        meta = AMReXBasicMeta(PLOTFILE_0)
        
        assert meta.n_fields > 0
        assert len(meta.field_list) == meta.n_fields
        assert meta.dimensionality in [2, 3]
        assert meta.max_level >= 0
        assert len(meta.domain_left_edge) == meta.dimensionality
        assert len(meta.domain_right_edge) == meta.dimensionality
    
    def test_multi_grid_meta(self):
        """Test multi-grid metadata across files."""
        from xamrex.metadata import AMReXMultiGridMeta
        
        meta = AMReXMultiGridMeta([PLOTFILE_0, PLOTFILE_360])
        
        assert len(meta.plotfile_paths) == 2
        assert len(meta.time_values) == 2
        assert meta.max_level_ever >= 0
        assert len(meta.all_grids) > 0
        assert len(meta.level_availability) > 0
    
    def test_level_availability_tracking(self):
        """Test that level availability is tracked correctly."""
        from xamrex.metadata import AMReXMultiGridMeta
        
        meta = AMReXMultiGridMeta(ALL_PLOTFILES)
        
        # Level 0 should always be available
        assert 0 in meta.level_availability
        assert len(meta.level_availability[0]) == len(ALL_PLOTFILES)
        
        # Check method
        for level in meta.level_availability:
            for time_idx in range(len(ALL_PLOTFILES)):
                available = meta.is_level_available(level, time_idx)
                assert isinstance(available, bool)


class TestCoordinates:
    """Test C-grid coordinate generation."""
    
    def test_coordinate_generation_rho(self):
        """Test coordinate generation for rho-points."""
        from xamrex.metadata import AMReXBasicMeta
        from xamrex.coordinates import CGridCoordinateGenerator
        
        meta = AMReXBasicMeta(PLOTFILE_0)
        coord_gen = CGridCoordinateGenerator(
            meta.domain_left_edge,
            meta.domain_right_edge,
            meta.level_dimensions,
            meta.dimensionality
        )
        
        coords = coord_gen.generate_coordinates(0, 'rho', is_2d=False)
        
        assert 'x_rho' in coords
        assert 'y_rho' in coords
        if meta.dimensionality == 3:
            assert 'z_rho' in coords
        
        # Check coordinate structure
        for coord_name, (dim_name, array, attrs) in coords.items():
            assert isinstance(array, np.ndarray)
            assert 'axis' in attrs
            assert 'c_grid_axis_shift' in attrs
    
    def test_staggered_coordinates(self):
        """Test that u/v/w coordinates are properly staggered."""
        from xamrex.metadata import AMReXBasicMeta
        from xamrex.coordinates import CGridCoordinateGenerator
        
        meta = AMReXBasicMeta(PLOTFILE_0)
        coord_gen = CGridCoordinateGenerator(
            meta.domain_left_edge,
            meta.domain_right_edge,
            meta.level_dimensions,
            meta.dimensionality
        )
        
        # Get coordinates for different grids
        rho_coords = coord_gen.generate_coordinates(0, 'rho', is_2d=False)
        u_coords = coord_gen.generate_coordinates(0, 'u', is_2d=False)
        
        # U-points should be shifted in x
        x_rho = rho_coords['x_rho'][1]
        x_u = u_coords['x_u'][1]
        
        # U-points at cell faces, rho at centers
        assert not np.allclose(x_rho, x_u)
        
        # Check shift attributes
        assert rho_coords['x_rho'][2]['c_grid_axis_shift'] == 0.0
        assert u_coords['x_u'][2]['c_grid_axis_shift'] == -0.5
    
    def test_dimension_names(self):
        """Test dimension name generation."""
        from xamrex.metadata import AMReXBasicMeta
        from xamrex.coordinates import CGridCoordinateGenerator
        
        meta = AMReXBasicMeta(PLOTFILE_0)
        coord_gen = CGridCoordinateGenerator(
            meta.domain_left_edge,
            meta.domain_right_edge,
            meta.level_dimensions,
            meta.dimensionality
        )
        
        # 3D variable
        dims_3d = coord_gen.get_dimension_names('rho', is_2d=False)
        assert dims_3d[0] == 'ocean_time'
        assert 'z_rho' in dims_3d
        assert 'y_rho' in dims_3d
        assert 'x_rho' in dims_3d
        
        # 2D variable
        dims_2d = coord_gen.get_dimension_names('rho', is_2d=True)
        assert len(dims_2d) == 3
        assert dims_2d[0] == 'ocean_time'
        assert 'z' not in str(dims_2d)


class TestFABLoader:
    """Test FAB data loading."""
    
    def test_fab_metadata_parsing(self):
        """Test FAB metadata parsing."""
        from xamrex.fab_loader import FABMetadata
        from xamrex.grid_detector import GridDetector
        
        detector = GridDetector()
        grids = detector.detect_grids(PLOTFILE_0, level=0)
        
        # Get first grid
        dir_name = list(grids.keys())[0]
        grid_info = grids[dir_name]
        
        fab_meta = FABMetadata(
            PLOTFILE_0, 0, dir_name,
            grid_info['num_components'],
            grid_info['dimensionality']
        )
        
        assert fab_meta.nfabs > 0
        assert len(fab_meta.metadata) == fab_meta.nfabs
        assert 'lo_i' in fab_meta.metadata.columns
        assert 'hi_i' in fab_meta.metadata.columns
    
    def test_masked_loader(self):
        """Test masked FAB loader for missing data."""
        from xamrex.fab_loader import MaskedFABLoader
        
        shape = (1, 10, 20, 30)
        loader = MaskedFABLoader(shape)
        dask_array = loader.create_dask_array()
        
        assert dask_array.shape == shape
        
        # Compute and check for NaN
        data = dask_array.compute()
        assert np.all(np.isnan(data))


class TestBackend:
    """Test xarray backend integration."""
    
    def test_single_file_loading(self):
        """Test loading a single plotfile."""
        import xarray as xr
        
        ds = xr.open_dataset(
            PLOTFILE_0,
            engine='xamrex',
            level=0
        )
        
        assert isinstance(ds, xr.Dataset)
        assert 'ocean_time' in ds.coords
        assert len(ds.data_vars) > 0
    
    def test_time_series_loading(self):
        """Test loading multiple plotfiles as time series."""
        import xarray as xr
        
        ds = xr.open_dataset(
            [PLOTFILE_0, PLOTFILE_360],
            engine='xamrex',
            level=0
        )
        
        assert isinstance(ds, xr.Dataset)
        assert 'ocean_time' in ds.coords
        assert len(ds.ocean_time) == 2
    
    def test_xgcm_metadata(self):
        """Test that xgcm-compatible metadata is present."""
        import xarray as xr
        
        ds = xr.open_dataset(PLOTFILE_0, engine='xamrex', level=0)
        
        assert 'xgcm-Grid' in ds.attrs
        grid_spec = ds.attrs['xgcm-Grid']
        assert isinstance(grid_spec, dict)
    
    def test_coordinate_attributes(self):
        """Test that coordinates have proper attributes."""
        import xarray as xr
        
        ds = xr.open_dataset(PLOTFILE_0, engine='xamrex', level=0)
        
        # Check rho coordinates
        if 'x_rho' in ds.coords:
            assert 'axis' in ds.x_rho.attrs
            assert 'c_grid_axis_shift' in ds.x_rho.attrs
        
        # Check u coordinates if present
        if 'x_u' in ds.coords:
            assert 'axis' in ds.x_u.attrs
            assert ds.x_u.attrs['c_grid_axis_shift'] == -0.5
    
    def test_lazy_loading(self):
        """Test that data is loaded lazily."""
        import xarray as xr
        import dask.array as da
        
        ds = xr.open_dataset(PLOTFILE_0, engine='xamrex', level=0)
        
        # Data should be dask arrays
        for var_name in ds.data_vars:
            assert isinstance(ds[var_name].data, da.Array)
    
    def test_drop_variables(self):
        """Test dropping variables."""
        import xarray as xr
        
        ds_full = xr.open_dataset(PLOTFILE_0, engine='xamrex', level=0)
        var_to_drop = list(ds_full.data_vars)[0]
        
        ds_dropped = xr.open_dataset(
            PLOTFILE_0,
            engine='xamrex',
            level=0,
            drop_variables=var_to_drop
        )
        
        assert var_to_drop not in ds_dropped.data_vars
        assert len(ds_dropped.data_vars) == len(ds_full.data_vars) - 1


class TestIntegration:
    """Integration tests with ocean_out data."""
    
    def test_detect_all_grids(self):
        """Test that all expected grids are detected."""
        import xarray as xr
        
        ds = xr.open_dataset(PLOTFILE_0, engine='xamrex', level=0)
        
        # Check for various coordinate types
        coord_types = set()
        for coord in ds.coords:
            if '_' in str(coord):
                grid_type = str(coord).split('_')[1]
                coord_types.add(grid_type)
        
        assert len(coord_types) > 0
    
    def test_variable_grid_mapping(self):
        """Test that variables are on correct grids."""
        import xarray as xr
        
        ds = xr.open_dataset(PLOTFILE_0, engine='xamrex', level=0)
        
        # Check variable attributes
        for var_name in ds.data_vars:
            assert 'grid' in ds[var_name].attrs
            assert 'directory' in ds[var_name].attrs
    
    def test_time_series_shapes(self):
        """Test that time series have consistent spatial shapes."""
        import xarray as xr
        
        ds = xr.open_dataset(ALL_PLOTFILES[:2], engine='xamrex', level=0)
        
        for var_name in ds.data_vars:
            var = ds[var_name]
            # First dimension should be time
            assert var.dims[0] == 'ocean_time'
            assert var.shape[0] == 2  # Two timesteps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
