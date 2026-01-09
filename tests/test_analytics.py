"""Tests for Analytics module."""

import numpy as np
import pytest
from Analytics.distributions import normal_cdf, normal_pdf


class TestNormalCDF:
    """Test cases for normal_cdf function."""
    
    def test_at_zero(self):
        """Test that N(0) = 0.5."""
        assert abs(normal_cdf(0.0) - 0.5) < 1e-10
    
    def test_symmetry(self):
        """Test that N(-x) = 1 - N(x)."""
        x = 1.5
        assert abs(normal_cdf(-x) - (1.0 - normal_cdf(x))) < 1e-10
    
    def test_known_values(self):
        """Test against known values."""
        # N(1.96) ≈ 0.975
        assert abs(normal_cdf(1.96) - 0.975) < 0.001
        # N(-1.96) ≈ 0.025
        assert abs(normal_cdf(-1.96) - 0.025) < 0.001
    
    def test_array_input(self):
        """Test with array input."""
        x = np.array([0.0, 1.0, -1.0])
        result = normal_cdf(x)
        assert result.shape == x.shape
        assert abs(result[0] - 0.5) < 1e-10


class TestNormalPDF:
    """Test cases for normal_pdf function."""
    
    def test_at_zero(self):
        """Test that φ(0) = 1/√(2π)."""
        expected = 1.0 / np.sqrt(2.0 * np.pi)
        assert abs(normal_pdf(0.0) - expected) < 1e-10
    
    def test_symmetry(self):
        """Test that φ(-x) = φ(x)."""
        x = 1.5
        assert abs(normal_pdf(-x) - normal_pdf(x)) < 1e-10
    
    def test_positive(self):
        """Test that PDF is always positive."""
        x = np.linspace(-5, 5, 11)
        result = normal_pdf(x)
        assert np.all(result > 0)
    
    def test_decreasing_from_zero(self):
        """Test that PDF decreases as we move away from zero."""
        assert normal_pdf(0.0) > normal_pdf(1.0)
        assert normal_pdf(1.0) > normal_pdf(2.0)
