"""
Comprehensive test suite for xamrex package.
Tests critical components including backend functionality, data loading, and utilities.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import xarray as xr
from pathlib import Path

# Import xamrex components
import xamrex
from xamrex import AMReXSingleLevelEntrypoint
from xamrex.AMReX_array import AMReXDatasetMeta, AMReXFabsMetaSingleLevel

class TestXamrex:
    """Test suite for xamrex package."""
    
    def __init__(self):
        """Set up test data paths."""
        self.test_data_dir = Path(__file__).parent.parent / "test_data"
        self.test_files = list(self.test_data_dir.glob("plt_ml_quad*"))
        if not self.test_files:
            raise FileNotFoundError(f"No test data found in {self.test_data_dir}")
        self.test_file = self.test_files[0]  # Use first available test file
    
    def test_package_import(self):
        """Test that xamrex package imports correctly."""
        assert xamrex.__version__ is not None
        assert hasattr(xamrex, 'AMReXSingleLevelEntrypoint')
        assert hasattr(xamrex, 'open_amrex_levels')
        print("✓ Package import test passed")
    
    def test_backend_entrypoint(self):
        """Test AMReX backend entrypoint registration."""
        backend = AMReXSingleLevelEntrypoint()
        assert backend.guess_can_open(self.test_file)
        print("✓ Backend entrypoint test passed")
    
    def test_xarray_backend_integration(self):
        """Test xarray backend integration."""
        # Test opening with xarray using the backend
        ds = xr.open_dataset(self.test_file, engine='amrex', level=0)
        
        # Verify dataset structure
        assert isinstance(ds, xr.Dataset)
        assert 'temp' in ds.data_vars or 'salt' in ds.data_vars  # Should have at least one field
        assert 'x' in ds.coords
        assert 'y' in ds.coords
        
        # Verify attributes
        assert 'level' in ds.attrs
        assert ds.attrs['level'] == 0
        
        print("✓ xarray backend integration test passed")
    
    def test_amrex_metadata_parsing(self):
        """Test AMReX metadata parsing."""
        meta = AMReXDatasetMeta(self.test_file)
        
        # Verify metadata structure
        assert meta.dimensionality in [2, 3]
        assert len(meta.field_list) > 0
        assert meta.max_level >= 0
        assert meta.current_time >= 0
        assert len(meta.domain_left_edge) == meta.dimensionality
        assert len(meta.domain_right_edge) == meta.dimensionality
        
        print("✓ AMReX metadata parsing test passed")
    
    def test_fab_metadata_parsing(self):
        """Test FAB metadata parsing."""
        meta = AMReXDatasetMeta(self.test_file)
        fab_meta = AMReXFabsMetaSingleLevel(self.test_file, meta.n_fields, meta.dimensionality, level=0)
        
        # Verify FAB metadata structure
        assert not fab_meta.metadata.empty
        assert 'filename' in fab_meta.metadata.columns
        assert 'byte_offset' in fab_meta.metadata.columns
        assert 'lo_i' in fab_meta.metadata.columns
        assert 'hi_i' in fab_meta.metadata.columns
        
        print("✓ FAB metadata parsing test passed")
    
    def test_data_loading_lazy(self):
        """Test lazy data loading."""
        ds = xr.open_dataset(self.test_file, engine='amrex', level=0)
        
        # Get first data variable
        var_name = list(ds.data_vars.keys())[0]
        data_var = ds[var_name]
        
        # Verify lazy loading (data should be dask array)
        assert hasattr(data_var.data, 'compute')  # Should be a dask array
        
        # Test actual data access
        computed_data = data_var.compute()
        assert isinstance(computed_data.data, np.ndarray)
        assert computed_data.shape == data_var.shape
        
        print("✓ Lazy data loading test passed")
    
    def test_coordinate_calculation(self):
        """Test coordinate calculation for different levels."""
        # Test level 0
        ds0 = xr.open_dataset(self.test_file, engine='amrex', level=0)
        
        # Verify coordinates exist and are reasonable
        assert 'x' in ds0.coords
        assert 'y' in ds0.coords
        
        x_coord = ds0.coords['x']
        y_coord = ds0.coords['y']
        
        # Check coordinate properties
        assert len(x_coord) > 0
        assert len(y_coord) > 0
        assert np.all(np.diff(x_coord) > 0)  # Should be monotonically increasing
        assert np.all(np.diff(y_coord) > 0)  # Should be monotonically increasing
        
        print("✓ Coordinate calculation test passed")
    
    def test_multilevel_access(self):
        """Test accessing different AMR levels."""
        meta = AMReXDatasetMeta(self.test_file)
        
        # Test level 0
        ds0 = xr.open_dataset(self.test_file, engine='amrex', level=0)
        assert ds0.attrs['level'] == 0
        
        # Test level 1 if it exists
        if meta.max_level >= 1:
            ds1 = xr.open_dataset(self.test_file, engine='amrex', level=1)
            assert ds1.attrs['level'] == 1
            
            # Level 1 should have higher resolution
            assert len(ds1.coords['x']) >= len(ds0.coords['x'])
            assert len(ds1.coords['y']) >= len(ds0.coords['y'])
        
        print("✓ Multilevel access test passed")
    
    def test_utilities_functions(self):
        """Test utility functions."""
        # Test open_amrex_levels
        levels_dict = xamrex.open_amrex_levels(self.test_file)
        assert isinstance(levels_dict, dict)
        assert 0 in levels_dict
        
        # Test get_available_levels_from_file
        available_levels = xamrex.get_available_levels_from_file(self.test_file)
        assert isinstance(available_levels, list)
        assert 0 in available_levels
        
        # Test get_max_level
        max_level = xamrex.get_max_level(self.test_file)
        assert isinstance(max_level, int)
        assert max_level >= 0
        
        print("✓ Utility functions test passed")
    
    def test_data_integrity(self):
        """Test data integrity and reasonable values."""
        ds = xr.open_dataset(self.test_file, engine='amrex', level=0)
        
        # Get first data variable
        var_name = list(ds.data_vars.keys())[0]
        data_var = ds[var_name]
        
        # Compute a small subset of data
        subset = data_var.isel(x=slice(0, 10), y=slice(0, 10)).compute()
        
        # Check for reasonable values (not all NaN, not all zeros)
        valid_data = subset.data[~np.isnan(subset.data)]
        if len(valid_data) > 0:
            assert not np.all(valid_data == 0), "Data should not be all zeros"
            assert np.all(np.isfinite(valid_data)), "Data should be finite"
        
        print("✓ Data integrity test passed")

def run_tests():
    """Run all tests."""
    print("Running xamrex test suite...")
    print("=" * 50)
    
    try:
        test_instance = TestXamrex()
        
        # Run all test methods
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        passed = 0
        failed = 0
        
        for method_name in test_methods:
            try:
                method = getattr(test_instance, method_name)
                method()
                passed += 1
            except Exception as e:
                print(f"✗ {method_name} failed: {e}")
                failed += 1
        
        print("=" * 50)
        print(f"Tests completed: {passed} passed, {failed} failed")
        
        if failed == 0:
            print("🎉 All tests passed!")
            return True
        else:
            print("❌ Some tests failed!")
            return False
            
    except Exception as e:
        print(f"Test setup failed: {e}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
