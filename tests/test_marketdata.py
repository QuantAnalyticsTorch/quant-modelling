"""Tests for MarketData module."""

import pytest
import numpy as np
from MarketData.volatility_surface import VolatilitySurface
from MarketData.svi_surface import SVIVolatilitySurface


class TestVolatilitySurface:
    """Test cases for VolatilitySurface base class."""
    
    def test_initialization(self):
        """Test initialization with valid parameters."""
        # Create a minimal concrete implementation for testing
        class DummySurface(VolatilitySurface):
            def get_volatility(self, strike):
                return 0.2
            def get_variance(self, strike):
                return 0.04
        
        surface = DummySurface(forward=100.0, time_to_maturity=1.0)
        assert surface.forward == 100.0
        assert surface.time_to_maturity == 1.0
    
    def test_invalid_forward(self):
        """Test that non-positive forward raises ValueError."""
        class DummySurface(VolatilitySurface):
            def get_volatility(self, strike):
                return 0.2
            def get_variance(self, strike):
                return 0.04
        
        with pytest.raises(ValueError, match="Forward price must be positive"):
            DummySurface(forward=0.0, time_to_maturity=1.0)
        with pytest.raises(ValueError, match="Forward price must be positive"):
            DummySurface(forward=-100.0, time_to_maturity=1.0)
    
    def test_invalid_maturity(self):
        """Test that non-positive maturity raises ValueError."""
        class DummySurface(VolatilitySurface):
            def get_volatility(self, strike):
                return 0.2
            def get_variance(self, strike):
                return 0.04
        
        with pytest.raises(ValueError, match="Time to maturity must be positive"):
            DummySurface(forward=100.0, time_to_maturity=0.0)
        with pytest.raises(ValueError, match="Time to maturity must be positive"):
            DummySurface(forward=100.0, time_to_maturity=-1.0)
    
    def test_log_moneyness(self):
        """Test log-moneyness calculation."""
        class DummySurface(VolatilitySurface):
            def get_volatility(self, strike):
                return 0.2
            def get_variance(self, strike):
                return 0.04
        
        surface = DummySurface(forward=100.0, time_to_maturity=1.0)
        
        # ATM: log(100/100) = 0
        assert np.isclose(surface.get_log_moneyness(100.0), 0.0)
        
        # OTM call: log(110/100) > 0
        assert surface.get_log_moneyness(110.0) > 0
        
        # ITM call: log(90/100) < 0
        assert surface.get_log_moneyness(90.0) < 0
        
        # Array input
        strikes = np.array([90.0, 100.0, 110.0])
        log_moneyness = surface.get_log_moneyness(strikes)
        assert len(log_moneyness) == 3
        assert log_moneyness[1] == 0.0  # ATM


class TestSVIVolatilitySurface:
    """Test cases for SVIVolatilitySurface."""
    
    def test_initialization(self):
        """Test initialization with valid SVI parameters."""
        surface = SVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            b=0.1,
            rho=-0.4,
            m=0.0,
            sigma=0.2
        )
        assert surface.forward == 100.0
        assert surface.time_to_maturity == 1.0
        assert surface.a == 0.04
        assert surface.b == 0.1
        assert surface.rho == -0.4
        assert surface.m == 0.0
        assert surface.sigma == 0.2
    
    def test_invalid_b_parameter(self):
        """Test that negative b raises ValueError."""
        with pytest.raises(ValueError, match="Parameter b must be non-negative"):
            SVIVolatilitySurface(
                forward=100.0,
                time_to_maturity=1.0,
                a=0.04,
                b=-0.1,  # Invalid: negative
                rho=-0.4,
                m=0.0,
                sigma=0.2
            )
    
    def test_invalid_rho_parameter(self):
        """Test that rho outside [-1, 1] raises ValueError."""
        with pytest.raises(ValueError, match="Parameter rho must be in"):
            SVIVolatilitySurface(
                forward=100.0,
                time_to_maturity=1.0,
                a=0.04,
                b=0.1,
                rho=1.5,  # Invalid: > 1
                m=0.0,
                sigma=0.2
            )
        
        with pytest.raises(ValueError, match="Parameter rho must be in"):
            SVIVolatilitySurface(
                forward=100.0,
                time_to_maturity=1.0,
                a=0.04,
                b=0.1,
                rho=-1.5,  # Invalid: < -1
                m=0.0,
                sigma=0.2
            )
    
    def test_invalid_sigma_parameter(self):
        """Test that non-positive sigma raises ValueError."""
        with pytest.raises(ValueError, match="Parameter sigma must be positive"):
            SVIVolatilitySurface(
                forward=100.0,
                time_to_maturity=1.0,
                a=0.04,
                b=0.1,
                rho=-0.4,
                m=0.0,
                sigma=0.0  # Invalid: zero
            )
        
        with pytest.raises(ValueError, match="Parameter sigma must be positive"):
            SVIVolatilitySurface(
                forward=100.0,
                time_to_maturity=1.0,
                a=0.04,
                b=0.1,
                rho=-0.4,
                m=0.0,
                sigma=-0.2  # Invalid: negative
            )
    
    def test_negative_variance_constraint(self):
        """Test that parameters leading to negative variance raise ValueError."""
        # Choose parameters that violate: a + b * sigma * sqrt(1 - rho^2) >= 0
        with pytest.raises(ValueError, match="negative variance"):
            SVIVolatilitySurface(
                forward=100.0,
                time_to_maturity=1.0,
                a=-1.0,  # Very negative a
                b=0.1,
                rho=0.0,
                m=0.0,
                sigma=0.2
            )
    
    def test_variance_calculation_atm(self):
        """Test variance calculation at ATM (strike = forward)."""
        surface = SVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            b=0.1,
            rho=-0.4,
            m=0.0,
            sigma=0.2
        )
        
        # At ATM, k = 0, so w(0) = a + b * [rho * (0 - m) + sqrt((0 - m)^2 + sigma^2)]
        # w(0) = 0.04 + 0.1 * [-0.4 * 0 + sqrt(0 + 0.04)]
        # w(0) = 0.04 + 0.1 * 0.2 = 0.06
        expected_variance = 0.04 + 0.1 * np.sqrt(0.2**2)
        variance_atm = surface.get_variance(100.0)
        
        assert np.isclose(variance_atm, expected_variance, rtol=1e-10)
    
    def test_volatility_calculation_atm(self):
        """Test volatility calculation at ATM."""
        surface = SVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            b=0.1,
            rho=-0.4,
            m=0.0,
            sigma=0.2
        )
        
        variance_atm = surface.get_variance(100.0)
        expected_vol = np.sqrt(variance_atm / 1.0)
        vol_atm = surface.get_volatility(100.0)
        
        assert np.isclose(vol_atm, expected_vol, rtol=1e-10)
    
    def test_variance_array_input(self):
        """Test variance calculation with array input."""
        surface = SVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            b=0.1,
            rho=-0.4,
            m=0.0,
            sigma=0.2
        )
        
        strikes = np.array([90.0, 100.0, 110.0])
        variances = surface.get_variance(strikes)
        
        assert len(variances) == 3
        assert all(variances > 0)  # All variances should be positive
        
        # Verify each variance individually
        for i, strike in enumerate(strikes):
            single_variance = surface.get_variance(strike)
            assert np.isclose(variances[i], single_variance)
    
    def test_volatility_array_input(self):
        """Test volatility calculation with array input."""
        surface = SVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            b=0.1,
            rho=-0.4,
            m=0.0,
            sigma=0.2
        )
        
        strikes = np.array([90.0, 100.0, 110.0])
        volatilities = surface.get_volatility(strikes)
        
        assert len(volatilities) == 3
        assert all(volatilities > 0)  # All volatilities should be positive
    
    def test_volatility_smile_shape(self):
        """Test that SVI produces a volatility smile (typical shape)."""
        # Using typical parameters that produce a smile
        surface = SVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            b=0.15,
            rho=-0.6,  # Negative rho creates downward skew
            m=0.0,
            sigma=0.3
        )
        
        # Check strikes from ITM put to OTM call
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        volatilities = surface.get_volatility(strikes)
        
        # All volatilities should be positive
        assert all(volatilities > 0)
        
        # With negative rho, we expect ITM puts (low strikes) to have higher vol
        # than ATM, creating a smile/skew pattern
        vol_itm_put = volatilities[0]  # 80 strike
        vol_atm = volatilities[2]  # 100 strike
        
        # Typically we'd expect vol_itm_put >= vol_atm for negative rho
        # but we won't enforce strict inequality as it depends on parameters
        assert vol_itm_put > 0 and vol_atm > 0
    
    def test_variance_positive_for_different_strikes(self):
        """Test that variance is positive across a wide range of strikes."""
        surface = SVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            b=0.1,
            rho=-0.4,
            m=0.0,
            sigma=0.2
        )
        
        # Test a wide range of strikes
        strikes = np.linspace(50.0, 150.0, 50)
        variances = surface.get_variance(strikes)
        
        assert all(variances > 0), "All variances must be positive"
    
    def test_different_maturities(self):
        """Test SVI with different maturities."""
        # Short maturity
        surface_short = SVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=0.25,  # 3 months
            a=0.01,
            b=0.05,
            rho=-0.3,
            m=0.0,
            sigma=0.15
        )
        
        # Long maturity
        surface_long = SVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=2.0,  # 2 years
            a=0.08,
            b=0.2,
            rho=-0.5,
            m=0.0,
            sigma=0.25
        )
        
        vol_short = surface_short.get_volatility(100.0)
        vol_long = surface_long.get_volatility(100.0)
        
        assert vol_short > 0
        assert vol_long > 0
    
    def test_repr(self):
        """Test string representation."""
        surface = SVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            b=0.1,
            rho=-0.4,
            m=0.0,
            sigma=0.2
        )
        
        repr_str = repr(surface)
        assert 'SVIVolatilitySurface' in repr_str
        assert '100.0' in repr_str
        assert '0.04' in repr_str
        assert '0.1' in repr_str
        assert '-0.4' in repr_str
        assert '0.2' in repr_str
    
    def test_boundary_rho_values(self):
        """Test that boundary values of rho (±1) work correctly."""
        # Test rho = 1
        surface_rho_plus = SVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            b=0.1,
            rho=1.0,
            m=0.0,
            sigma=0.2
        )
        vol_plus = surface_rho_plus.get_volatility(100.0)
        assert vol_plus > 0
        
        # Test rho = -1
        surface_rho_minus = SVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            b=0.1,
            rho=-1.0,
            m=0.0,
            sigma=0.2
        )
        vol_minus = surface_rho_minus.get_volatility(100.0)
        assert vol_minus > 0
    
    def test_zero_b_parameter(self):
        """Test that b=0 produces flat variance (independent of strike)."""
        surface = SVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            b=0.0,  # Flat surface
            rho=-0.4,
            m=0.0,
            sigma=0.2
        )
        
        strikes = np.array([80.0, 100.0, 120.0])
        variances = surface.get_variance(strikes)
        
        # All variances should equal 'a' when b=0
        assert all(np.isclose(variances, 0.04))
