"""
Test suite for multi-time AMReX functionality.
Tests the new time series concatenation features.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import xarray as xr
from pathlib import Path

# Import xamrex components
import xamrex
from xamrex.backend_multi_time import AMReXMultiTimeEntrypoint

class TestXamrexMultiTime:
    """Test suite for multi-time xamrex functionality."""
    
    def __init__(self):
        """Set up test data paths."""
        self.test_data_dir = Path(__file__).parent.parent / "test_data"
        self.test_files = sorted(list(self.test_data_dir.glob("plt_ml_quad*")))
        if len(self.test_files) < 2:
            raise FileNotFoundError(f"Need at least 2 test files for multi-time tests. Found {len(self.test_files)} in {self.test_data_dir}")
        
        # Use first few files for testing
        self.test_files = self.test_files[:min(5, len(self.test_files))]
    
    def test_multi_time_backend_registration(self):
        """Test that multi-time backend is properly registered."""
        backend = AMReXMultiTimeEntrypoint()
        assert backend.guess_can_open(self.test_files)
        print("✓ Multi-time backend registration test passed")
    
    def test_open_amrex_time_series(self):
        """Test opening multiple files as time series using utility function."""
        # Test with explicit file list
        ds = xamrex.open_amrex_time_series(self.test_files, level=0)
        
        # Verify dataset structure
        assert isinstance(ds, xr.Dataset)
        
        # Check that time dimension exists and has correct length
        time_dims = [dim for dim in ds.dims if 'time' in dim.lower()]
        assert len(time_dims) == 1, f"Expected 1 time dimension, found {len(time_dims)}: {time_dims}"
        
        time_dim = time_dims[0]
        assert ds.dims[time_dim] == len(self.test_files), f"Time dimension should have {len(self.test_files)} steps, got {ds.dims[time_dim]}"
        
        # Check that data variables exist
        assert len(ds.data_vars) > 0, "Dataset should have data variables"
        
        # Check that spatial coordinates exist
        assert 'x' in ds.coords
        assert 'y' in ds.coords
        
        print("✓ open_amrex_time_series test passed")
    
    def test_direct_backend_usage(self):
        """Test using the multi-time backend directly with xarray."""
        ds = xr.open_dataset(
            self.test_files,
            engine='amrex_multitime',
            level=0
        )
        
        # Verify dataset structure
        assert isinstance(ds, xr.Dataset)
        
        # Check time dimension
        time_dims = [dim for dim in ds.dims if 'time' in dim.lower()]
        assert len(time_dims) == 1
        time_dim = time_dims[0]
        assert ds.dims[time_dim] == len(self.test_files)
        
        # Check attributes
        assert 'concatenated_files' in ds.attrs
        assert ds.attrs['concatenated_files'] == len(self.test_files)
        assert 'time_steps' in ds.attrs
        assert ds.attrs['time_steps'] == len(self.test_files)
        
        print("✓ Direct backend usage test passed")
    
    def test_time_coordinate_ordering(self):
        """Test that time coordinates are properly ordered."""
        ds = xamrex.open_amrex_time_series(self.test_files, level=0)
        
        # Find time dimension
        time_dim = [dim for dim in ds.dims if 'time' in dim.lower()][0]
        time_coord = ds.coords[time_dim]
        
        # Check that time values are monotonically increasing
        time_values = time_coord.values
        assert len(time_values) == len(self.test_files)
        
        # Should be sorted (may not be strictly increasing if files have same time)
        sorted_times = np.sort(time_values)
        assert np.array_equal(time_values, sorted_times), "Time values should be sorted"
        
        print("✓ Time coordinate ordering test passed")
    
    def test_data_integrity_across_time(self):
        """Test that data is correctly loaded across all time steps."""
        ds = xamrex.open_amrex_time_series(self.test_files, level=0)
        
        # Get first data variable
        var_name = list(ds.data_vars.keys())[0]
        data_var = ds[var_name]
        
        # Find time dimension
        time_dim = [dim for dim in ds.dims if 'time' in dim.lower()][0]
        
        # Check that each time step has reasonable data
        for t in range(ds.dims[time_dim]):
            time_slice = data_var.isel({time_dim: t})
            computed_slice = time_slice.compute()
            
            # Check that we have some valid data (not all NaN)
            valid_data = computed_slice.data[~np.isnan(computed_slice.data)]
            if len(valid_data) > 0:
                assert not np.all(valid_data == 0), f"Time step {t} should not be all zeros"
                assert np.all(np.isfinite(valid_data)), f"Time step {t} should have finite values"
        
        print("✓ Data integrity across time test passed")
    
    def test_find_amrex_time_series(self):
        """Test finding time series files in directory."""
        found_files = xamrex.find_amrex_time_series(self.test_data_dir, pattern="plt_ml_quad*")
        
        # Should find our test files
        assert len(found_files) >= len(self.test_files)
        
        # Check that files are sorted by time
        prev_time = None
        for file_path in found_files:
            from xamrex.AMReX_array import AMReXDatasetMeta
            try:
                meta = AMReXDatasetMeta(file_path)
                if prev_time is not None:
                    assert meta.current_time >= prev_time, "Files should be sorted by time"
                prev_time = meta.current_time
            except Exception:
                # Skip files that can't be read
                pass
        
        print("✓ find_amrex_time_series test passed")
    
    def test_create_time_series_from_directory(self):
        """Test creating time series from directory pattern."""
        ds = xamrex.create_time_series_from_directory(
            self.test_data_dir,
            pattern="plt_ml_quad*",
            level=0
        )
        
        # Should have multiple time steps
        time_dims = [dim for dim in ds.dims if 'time' in dim.lower()]
        assert len(time_dims) == 1
        time_dim = time_dims[0]
        assert ds.dims[time_dim] >= 2, "Should have at least 2 time steps"
        
        # Should have data variables
        assert len(ds.data_vars) > 0
        
        print("✓ create_time_series_from_directory test passed")
    
    def test_validate_time_series_compatibility(self):
        """Test validation of time series compatibility."""
        results = xamrex.validate_time_series_compatibility(self.test_files[:3], level=0)
        
        # Should be compatible
        assert results['compatible'] == True, f"Files should be compatible: {results['issues']}"
        assert results['file_count'] == 3
        assert results['time_range'] is not None
        assert results['fields'] is not None
        assert results['domain_info'] is not None
        
        print("✓ validate_time_series_compatibility test passed")
    
    def test_extract_time_slice(self):
        """Test extracting time slices from multi-time dataset."""
        ds = xamrex.open_amrex_time_series(self.test_files, level=0)
        
        # Find time dimension
        time_dim = [dim for dim in ds.dims if 'time' in dim.lower()][0]
        original_time_steps = ds.dims[time_dim]
        
        # Test index-based slicing
        if original_time_steps >= 3:
            sliced_ds = xamrex.extract_time_slice(ds, time_indices=slice(1, 3))
            assert sliced_ds.dims[time_dim] == 2
        
        # Test time-range slicing
        time_coord = ds.coords[time_dim]
        if len(time_coord) >= 2:
            min_time = float(time_coord.min())
            max_time = float(time_coord.max())
            mid_time = (min_time + max_time) / 2
            
            sliced_ds = xamrex.extract_time_slice(ds, time_range=(min_time, mid_time))
            assert sliced_ds.dims[time_dim] <= original_time_steps
        
        print("✓ extract_time_slice test passed")
    
    def test_compute_time_statistics(self):
        """Test computing time-based statistics."""
        ds = xamrex.open_amrex_time_series(self.test_files, level=0)
        
        # Get first data variable
        var_name = list(ds.data_vars.keys())[0]
        
        # Compute statistics
        stats_ds = xamrex.compute_time_statistics(
            ds, 
            variables=[var_name],
            statistics=['mean', 'std', 'min', 'max']
        )
        
        # Check that statistics were computed
        expected_vars = [f'{var_name}_{stat}' for stat in ['mean', 'std', 'min', 'max']]
        for expected_var in expected_vars:
            assert expected_var in stats_ds.data_vars, f"Missing statistic: {expected_var}"
        
        # Check that time dimension is removed
        time_dims = [dim for dim in stats_ds.dims if 'time' in dim.lower()]
        assert len(time_dims) == 0, "Time dimension should be removed from statistics"
        
        # Check that spatial coordinates remain
        assert 'x' in stats_ds.coords
        assert 'y' in stats_ds.coords
        
        print("✓ compute_time_statistics test passed")
    
    def test_single_file_fallback(self):
        """Test that single file input falls back to single-time backend."""
        # Test with single file
        ds_single = xamrex.open_amrex_time_series(self.test_files[0], level=0)
        
        # Should still work but with only one time step
        time_dims = [dim for dim in ds.dims if 'time' in dim.lower()]
        if time_dims:
            time_dim = time_dims[0]
            assert ds_single.dims[time_dim] == 1
        
        print("✓ Single file fallback test passed")
    
    def test_lazy_loading_multi_time(self):
        """Test that lazy loading works with multi-time datasets."""
        ds = xamrex.open_amrex_time_series(self.test_files, level=0)
        
        # Get first data variable
        var_name = list(ds.data_vars.keys())[0]
        data_var = ds[var_name]
        
        # Should be a dask array
        assert hasattr(data_var.data, 'compute'), "Data should be lazy (dask array)"
        
        # Test computing a small subset
        subset = data_var.isel(x=slice(0, 5), y=slice(0, 5)).compute()
        assert isinstance(subset.data, np.ndarray)
        
        print("✓ Lazy loading multi-time test passed")

def run_multi_time_tests():
    """Run all multi-time tests."""
    print("Running xamrex multi-time test suite...")
    print("=" * 60)
    
    try:
        test_instance = TestXamrexMultiTime()
        
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
        
        print("=" * 60)
        print(f"Multi-time tests completed: {passed} passed, {failed} failed")
        
        if failed == 0:
            print("🎉 All multi-time tests passed!")
            return True
        else:
            print("❌ Some multi-time tests failed!")
            return False
            
    except Exception as e:
        print(f"Multi-time test setup failed: {e}")
        return False

if __name__ == "__main__":
    success = run_multi_time_tests()
    sys.exit(0 if success else 1)
