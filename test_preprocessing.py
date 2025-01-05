import pytest
import numpy as np
import pandas as pd
from preprocessing import SignalPreprocessor

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = {
        'apr': [10.0, 20.0, 15.0, 25.0],
        'signal_amount': [100.0, 200.0, 150.0, 250.0],
        'weekly_queries': [1000, 2000, 1500, 2500],
        'total_earnings': [50.0, 100.0, 75.0, 125.0]
    }
    return pd.DataFrame(data)

def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    window_size = 5
    preprocessor = SignalPreprocessor(window_size=window_size)
    assert preprocessor.window_size == window_size
    assert preprocessor.feature_means is None
    assert preprocessor.feature_stds is None
    assert preprocessor.optimal_coefficients is None

def test_mean_reversion_time_calculation(sample_data):
    """Test mean reversion time calculation."""
    preprocessor = SignalPreprocessor()
    coefficients = np.array([0.5, 0.3, 0.2])
    
    reversion_time = preprocessor.calculate_mean_reversion_time(sample_data, coefficients)
    assert isinstance(reversion_time, float)
    assert reversion_time > 0 or reversion_time == float('inf')

def test_coefficient_optimization(sample_data):
    """Test coefficient optimization."""
    preprocessor = SignalPreprocessor()
    coefficients = preprocessor.optimize_coefficients(sample_data)
    
    # Check coefficient properties
    assert isinstance(coefficients, np.ndarray)
    assert len(coefficients) == 3  # apr, signal_amount, weekly_queries
    assert np.all(coefficients >= 0)  # Non-negative coefficients
    assert np.abs(np.sum(np.abs(coefficients)) - 1.0) < 1e-6  # Sum to 1

def test_trend_feature_calculation(sample_data):
    """Test trend feature calculation."""
    preprocessor = SignalPreprocessor(window_size=2)
    result = preprocessor.calculate_trend_features(sample_data)
    
    # Check that trend features are added
    assert 'apr_ma' in result.columns
    assert 'apr_std' in result.columns
    assert 'apr_trend' in result.columns
    assert 'apr_momentum' in result.columns
    assert 'signal_momentum' in result.columns
    
    # Check calculations
    assert len(result) == len(sample_data)
    assert not result['apr_ma'].isna().any()
    assert not result['apr_std'].isna().any()

def test_fit_transform(sample_data):
    """Test complete preprocessing pipeline."""
    preprocessor = SignalPreprocessor()
    features = preprocessor.fit_transform(sample_data)
    
    # Check output properties
    assert isinstance(features, np.ndarray)
    assert features.dtype == np.float32
    assert len(features.shape) == 2
    
    # Check normalization
    assert np.abs(features.mean()) < 1.0  # Roughly centered
    assert np.abs(features.std() - 1.0) < 1.0  # Roughly unit variance

def test_transform_without_fit():
    """Test transform behavior without fitting."""
    preprocessor = SignalPreprocessor()
    data = pd.DataFrame({
        'apr': [10.0],
        'signal_amount': [100.0],
        'weekly_queries': [1000],
        'total_earnings': [50.0]
    })
    
    # Should handle single datapoint gracefully
    features = preprocessor.transform(data)
    assert isinstance(features, np.ndarray)
    assert features.dtype == np.float32

def test_feature_consistency(sample_data):
    """Test consistency of feature generation."""
    preprocessor = SignalPreprocessor()
    
    # First transform
    features1 = preprocessor.fit_transform(sample_data)
    
    # Second transform with same data
    features2 = preprocessor.transform(sample_data)
    
    # Features should be identical for same input
    np.testing.assert_array_almost_equal(features1, features2)

if __name__ == "__main__":
    pytest.main([__file__])
