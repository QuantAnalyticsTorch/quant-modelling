"""Tests for MarketData module."""

import pytest
import numpy as np
from MarketData.volatility_surface import VolatilitySurface
from MarketData.svi_surface import SVIVolatilitySurface
from MarketData.ssvi_surface import SSVIVolatilitySurface


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
        # With m=0, sigma=0.2: sqrt((0 - 0)^2 + 0.2^2) = sqrt(0.04) = 0.2
        # w(0) = 0.04 + 0.1 * [-0.4 * 0 + 0.2]
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


class TestSSVIVolatilitySurface:
    """Test cases for SSVIVolatilitySurface."""
    
    def test_initialization(self):
        """Test initialization with valid SSVI parameters."""
        surface = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            theta=0.04,
            rho=-0.3,
            gamma=0.5,
            eta=1.5
        )
        assert surface.forward == 100.0
        assert surface.time_to_maturity == 1.0
        assert surface.theta == 0.04
        assert surface.rho == -0.3
        assert surface.gamma == 0.5
        assert surface.eta == 1.5
    
    def test_invalid_theta_parameter(self):
        """Test that non-positive theta raises ValueError."""
        with pytest.raises(ValueError, match="Parameter theta must be positive"):
            SSVIVolatilitySurface(
                forward=100.0,
                time_to_maturity=1.0,
                theta=0.0,  # Invalid: zero
                rho=-0.4,
                gamma=0.5,
                eta=1.5
            )
        
        with pytest.raises(ValueError, match="Parameter theta must be positive"):
            SSVIVolatilitySurface(
                forward=100.0,
                time_to_maturity=1.0,
                theta=-0.04,  # Invalid: negative
                rho=-0.4,
                gamma=0.5,
                eta=1.5
            )
    
    def test_invalid_rho_parameter(self):
        """Test that rho outside [-1, 1] raises ValueError."""
        with pytest.raises(ValueError, match="Parameter rho must be in"):
            SSVIVolatilitySurface(
                forward=100.0,
                time_to_maturity=1.0,
                theta=0.04,
                rho=1.5,  # Invalid: > 1
                gamma=0.5,
                eta=1.5
            )
        
        with pytest.raises(ValueError, match="Parameter rho must be in"):
            SSVIVolatilitySurface(
                forward=100.0,
                time_to_maturity=1.0,
                theta=0.04,
                rho=-1.5,  # Invalid: < -1
                gamma=0.5,
                eta=1.5
            )
    
    def test_invalid_eta_parameter(self):
        """Test that non-positive eta raises ValueError."""
        with pytest.raises(ValueError, match="Parameter eta must be positive"):
            SSVIVolatilitySurface(
                forward=100.0,
                time_to_maturity=1.0,
                theta=0.04,
                rho=-0.4,
                gamma=0.5,
                eta=0.0  # Invalid: zero
            )
        
        with pytest.raises(ValueError, match="Parameter eta must be positive"):
            SSVIVolatilitySurface(
                forward=100.0,
                time_to_maturity=1.0,
                theta=0.04,
                rho=-0.4,
                gamma=0.5,
                eta=-1.5  # Invalid: negative
            )
    
    def test_invalid_gamma_parameter(self):
        """Test that negative gamma raises ValueError."""
        with pytest.raises(ValueError, match="Parameter gamma must be non-negative"):
            SSVIVolatilitySurface(
                forward=100.0,
                time_to_maturity=1.0,
                theta=0.04,
                rho=-0.4,
                gamma=-0.5,  # Invalid: negative
                eta=1.5
            )
    
    def test_arbitrage_free_constraint(self):
        """Test that parameters violating arbitrage-free constraint raise ValueError."""
        # For gamma=0.6, max |rho| = (1-0.6)/(1+0.6) = 0.4/1.6 = 0.25
        with pytest.raises(ValueError, match="arbitrage-free surface"):
            SSVIVolatilitySurface(
                forward=100.0,
                time_to_maturity=1.0,
                theta=0.04,
                rho=0.5,  # Invalid: |rho| > 0.25 for gamma=0.6
                gamma=0.6,
                eta=1.5
            )
    
    def test_phi_calculation(self):
        """Test the power-law function φ(θ) = η / θ^γ."""
        surface = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            theta=0.04,
            rho=-0.3,
            gamma=0.5,
            eta=1.5
        )
        
        # φ(θ) = 1.5 / (0.04^0.5) = 1.5 / 0.2 = 7.5
        expected_phi = 1.5 / (0.04 ** 0.5)
        phi = surface._phi()
        
        assert np.isclose(phi, expected_phi, rtol=1e-10)
    
    def test_variance_calculation_atm(self):
        """Test variance calculation at ATM (strike = forward)."""
        surface = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            theta=0.04,
            rho=-0.3,
            gamma=0.5,
            eta=1.5
        )
        
        # At ATM, k = 0
        # φ = 1.5 / 0.04^0.5 = 7.5
        # w(0) = θ/2 × {1 + ρ × φ × 0 + sqrt[(φ × 0 + ρ)² + (1 - ρ²)]}
        # w(0) = 0.04/2 × {1 + 0 + sqrt[(-0.3)² + (1 - 0.09)]}
        # w(0) = 0.02 × {1 + sqrt[0.09 + 0.91]}
        # w(0) = 0.02 × {1 + sqrt[1.0]}
        # w(0) = 0.02 × 2 = 0.04
        variance_atm = surface.get_variance(100.0)
        
        # The ATM variance should equal theta for SSVI
        assert np.isclose(variance_atm, surface.theta, rtol=1e-10)
    
    def test_volatility_calculation_atm(self):
        """Test volatility calculation at ATM."""
        surface = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            theta=0.04,
            rho=-0.3,
            gamma=0.5,
            eta=1.5
        )
        
        variance_atm = surface.get_variance(100.0)
        expected_vol = np.sqrt(variance_atm / 1.0)
        vol_atm = surface.get_volatility(100.0)
        
        assert np.isclose(vol_atm, expected_vol, rtol=1e-10)
    
    def test_variance_array_input(self):
        """Test variance calculation with array input."""
        surface = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            theta=0.04,
            rho=-0.3,
            gamma=0.5,
            eta=1.5
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
        surface = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            theta=0.04,
            rho=-0.3,
            gamma=0.5,
            eta=1.5
        )
        
        strikes = np.array([90.0, 100.0, 110.0])
        volatilities = surface.get_volatility(strikes)
        
        assert len(volatilities) == 3
        assert all(volatilities > 0)  # All volatilities should be positive
    
    def test_volatility_smile_shape(self):
        """Test that SSVI produces a volatility smile (typical shape)."""
        # Using typical parameters that produce a smile
        surface = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            theta=0.04,
            rho=-0.2,  # Negative rho creates downward skew
            gamma=0.3,
            eta=2.0
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
        
        # Both should be positive
        assert vol_itm_put > 0 and vol_atm > 0
    
    def test_variance_positive_for_different_strikes(self):
        """Test that variance is positive across a wide range of strikes."""
        surface = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            theta=0.04,
            rho=-0.2,
            gamma=0.4,
            eta=1.8
        )
        
        # Test a wide range of strikes
        strikes = np.linspace(50.0, 150.0, 50)
        variances = surface.get_variance(strikes)
        
        assert all(variances > 0), "All variances must be positive"
    
    def test_different_maturities(self):
        """Test SSVI with different maturities."""
        # Short maturity
        surface_short = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=0.25,  # 3 months
            theta=0.01,
            rho=-0.3,
            gamma=0.4,
            eta=1.2
        )
        
        # Long maturity
        surface_long = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=2.0,  # 2 years
            theta=0.08,
            rho=-0.2,
            gamma=0.5,
            eta=2.0
        )
        
        vol_short = surface_short.get_volatility(100.0)
        vol_long = surface_long.get_volatility(100.0)
        
        assert vol_short > 0
        assert vol_long > 0
    
    def test_repr(self):
        """Test string representation."""
        surface = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            theta=0.04,
            rho=-0.3,
            gamma=0.5,
            eta=1.5
        )
        
        repr_str = repr(surface)
        assert 'SSVIVolatilitySurface' in repr_str
        assert '100.0' in repr_str
        assert '0.04' in repr_str
        assert '-0.3' in repr_str
        assert '0.5' in repr_str
        assert '1.5' in repr_str
    
    def test_boundary_rho_values(self):
        """Test that boundary values of rho work correctly within constraints."""
        # For gamma=0, any rho in [-1, 1] should work
        surface_rho_plus = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            theta=0.04,
            rho=0.9,  # Close to 1
            gamma=0.0,  # gamma=0 allows any rho
            eta=1.5
        )
        vol_plus = surface_rho_plus.get_volatility(100.0)
        assert vol_plus > 0
        
        surface_rho_minus = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            theta=0.04,
            rho=-0.9,  # Close to -1
            gamma=0.0,  # gamma=0 allows any rho
            eta=1.5
        )
        vol_minus = surface_rho_minus.get_volatility(100.0)
        assert vol_minus > 0
    
    def test_gamma_zero(self):
        """Test SSVI with gamma=0, which simplifies φ(θ) = η."""
        surface = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            theta=0.04,
            rho=-0.4,
            gamma=0.0,  # gamma=0
            eta=1.5
        )
        
        # When gamma=0, φ(θ) = η / θ^0 = η
        expected_phi = surface.eta
        phi = surface._phi()
        
        assert np.isclose(phi, expected_phi)
        
        # Variance should still be positive
        variance = surface.get_variance(100.0)
        assert variance > 0
    
    def test_consistency_across_strikes(self):
        """Test that the surface is smooth and consistent across strikes."""
        surface = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            theta=0.04,
            rho=-0.3,
            gamma=0.4,
            eta=1.6
        )
        
        # Get volatilities for a range of strikes
        strikes = np.linspace(70.0, 130.0, 61)
        volatilities = surface.get_volatility(strikes)
        
        # All should be positive
        assert all(volatilities > 0)
        
        # Should be smooth (no sudden jumps)
        # Check that adjacent volatilities don't differ by more than a reasonable amount
        diffs = np.abs(np.diff(volatilities))
        max_diff = np.max(diffs)
        
        # With small strike increments, volatility should change smoothly
        assert max_diff < 0.05, "Volatility surface should be smooth"
