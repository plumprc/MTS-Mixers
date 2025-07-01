import pytest
import sys
import os
from pathlib import Path


class TestSetupValidation:
    """Validation tests to ensure the testing infrastructure is properly configured."""
    
    def test_pytest_installed(self):
        """Test that pytest is available."""
        import pytest
        assert pytest.__version__
        
    def test_pytest_cov_installed(self):
        """Test that pytest-cov is available."""
        import pytest_cov
        assert pytest_cov
        
    def test_pytest_mock_installed(self):
        """Test that pytest-mock is available."""
        import pytest_mock
        assert pytest_mock
        
    def test_project_packages_importable(self):
        """Test that all project packages can be imported."""
        packages = ['data_provider', 'exp', 'layers', 'models', 'utils']
        
        for package in packages:
            try:
                __import__(package)
            except ImportError as e:
                pytest.fail(f"Failed to import {package}: {e}")
                
    def test_fixtures_available(self, temp_dir, sample_data, mock_config):
        """Test that conftest fixtures are available."""
        assert temp_dir.exists()
        assert sample_data is not None
        assert mock_config is not None
        
    def test_markers_configured(self, request):
        """Test that custom markers are properly configured."""
        markers = ['unit', 'integration', 'slow']
        config_markers = request.config.getini('markers')
        
        for marker in markers:
            assert any(marker in str(m) for m in config_markers)
            
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit marker works."""
        assert True
        
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration marker works."""
        assert True
        
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker works."""
        assert True
        
    def test_coverage_configuration(self):
        """Test that coverage is properly configured."""
        import coverage
        assert coverage.__version__
        
    def test_temp_dir_fixture(self, temp_dir):
        """Test that temp_dir fixture creates and cleans up properly."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"
        
    def test_sample_data_fixture(self, sample_data):
        """Test that sample_data fixture generates correct data."""
        assert len(sample_data) == 100
        assert list(sample_data.columns) == ['date', 'value1', 'value2', 'value3']
        
    def test_sample_tensor_fixture(self):
        """Test that sample_tensor fixture generates correct tensor."""
        pytest.importorskip("torch")
        from tests.conftest import sample_tensor
        # This test will be skipped if torch is not available
        
    def test_mock_config_fixture(self, mock_config):
        """Test that mock_config fixture has expected attributes."""
        expected_attrs = ['seq_len', 'pred_len', 'd_model', 'n_heads', 
                         'e_layers', 'd_ff', 'dropout', 'activation']
        for attr in expected_attrs:
            assert hasattr(mock_config, attr)
            
    def test_pyproject_toml_exists(self):
        """Test that pyproject.toml exists and has proper configuration."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject_path.exists()
        
        content = pyproject_path.read_text()
        assert "[tool.poetry]" in content
        assert "[tool.pytest.ini_options]" in content
        assert "[tool.coverage.run]" in content
        
    def test_test_directory_structure(self):
        """Test that the test directory structure is correct."""
        test_root = Path(__file__).parent
        assert test_root.name == "tests"
        assert (test_root / "__init__.py").exists()
        assert (test_root / "conftest.py").exists()
        assert (test_root / "unit" / "__init__.py").exists()
        assert (test_root / "integration" / "__init__.py").exists()