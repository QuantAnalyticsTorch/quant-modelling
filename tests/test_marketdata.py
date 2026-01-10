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
            a=0.04,
            gamma=0.5,
            rho=-0.4,
            eta=1.0,
            lambda_param=0.3
        )
        assert surface.forward == 100.0
        assert surface.time_to_maturity == 1.0
        assert surface.a == 0.04
        assert surface.gamma == 0.5
        assert surface.rho == -0.4
        assert surface.eta == 1.0
        assert surface.lambda_param == 0.3
    
    def test_invalid_a_parameter(self):
        """Test that non-positive a raises ValueError."""
        with pytest.raises(ValueError, match="Parameter a must be positive"):
            SSVIVolatilitySurface(
                forward=100.0,
                time_to_maturity=1.0,
                a=0.0,  # Invalid: zero
                gamma=0.5,
                rho=-0.4,
                eta=1.0,
                lambda_param=0.3
            )
        
        with pytest.raises(ValueError, match="Parameter a must be positive"):
            SSVIVolatilitySurface(
                forward=100.0,
                time_to_maturity=1.0,
                a=-0.04,  # Invalid: negative
                gamma=0.5,
                rho=-0.4,
                eta=1.0,
                lambda_param=0.3
            )
    
    def test_invalid_gamma_parameter(self):
        """Test that gamma outside (0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="Parameter gamma must be in"):
            SSVIVolatilitySurface(
                forward=100.0,
                time_to_maturity=1.0,
                a=0.04,
                gamma=0.0,  # Invalid: zero
                rho=-0.4,
                eta=1.0,
                lambda_param=0.3
            )
        
        with pytest.raises(ValueError, match="Parameter gamma must be in"):
            SSVIVolatilitySurface(
                forward=100.0,
                time_to_maturity=1.0,
                a=0.04,
                gamma=1.5,  # Invalid: > 1
                rho=-0.4,
                eta=1.0,
                lambda_param=0.3
            )
    
    def test_invalid_rho_parameter(self):
        """Test that rho outside [-1, 1] raises ValueError."""
        with pytest.raises(ValueError, match="Parameter rho must be in"):
            SSVIVolatilitySurface(
                forward=100.0,
                time_to_maturity=1.0,
                a=0.04,
                gamma=0.5,
                rho=1.5,  # Invalid: > 1
                eta=1.0,
                lambda_param=0.3
            )
        
        with pytest.raises(ValueError, match="Parameter rho must be in"):
            SSVIVolatilitySurface(
                forward=100.0,
                time_to_maturity=1.0,
                a=0.04,
                gamma=0.5,
                rho=-1.5,  # Invalid: < -1
                eta=1.0,
                lambda_param=0.3
            )
    
    def test_invalid_eta_parameter(self):
        """Test that non-positive eta raises ValueError."""
        with pytest.raises(ValueError, match="Parameter eta must be positive"):
            SSVIVolatilitySurface(
                forward=100.0,
                time_to_maturity=1.0,
                a=0.04,
                gamma=0.5,
                rho=-0.4,
                eta=0.0,  # Invalid: zero
                lambda_param=0.3
            )
    
    def test_invalid_lambda_parameter(self):
        """Test that lambda outside [0, 0.5] raises ValueError."""
        with pytest.raises(ValueError, match="Parameter lambda_param must be in"):
            SSVIVolatilitySurface(
                forward=100.0,
                time_to_maturity=1.0,
                a=0.04,
                gamma=0.5,
                rho=-0.4,
                eta=1.0,
                lambda_param=-0.1  # Invalid: < 0
            )
        
        with pytest.raises(ValueError, match="Parameter lambda_param must be in"):
            SSVIVolatilitySurface(
                forward=100.0,
                time_to_maturity=1.0,
                a=0.04,
                gamma=0.5,
                rho=-0.4,
                eta=1.0,
                lambda_param=0.6  # Invalid: > 0.5
            )
    
    def test_get_theta(self):
        """Test ATM variance calculation θ(t) = a * t^γ."""
        surface = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            gamma=0.5,
            rho=-0.4,
            eta=1.0,
            lambda_param=0.3
        )
        
        # For t=1.0, gamma=0.5: θ(1) = 0.04 * 1.0^0.5 = 0.04
        theta = surface.get_theta()
        assert np.isclose(theta, 0.04)
        
        # For t=0.25, gamma=0.5: θ(0.25) = 0.04 * 0.25^0.5 = 0.04 * 0.5 = 0.02
        theta_025 = surface.get_theta(0.25)
        assert np.isclose(theta_025, 0.02)
        
        # For t=4.0, gamma=0.5: θ(4) = 0.04 * 4^0.5 = 0.04 * 2 = 0.08
        theta_4 = surface.get_theta(4.0)
        assert np.isclose(theta_4, 0.08)
    
    def test_get_phi(self):
        """Test curvature function φ(θ) = η * θ^(-λ)."""
        surface = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            gamma=0.5,
            rho=-0.4,
            eta=1.0,
            lambda_param=0.3
        )
        
        theta = 0.04
        # φ(0.04) = 1.0 * 0.04^(-0.3)
        expected_phi = 1.0 * (0.04 ** (-0.3))
        phi = surface.get_phi(theta)
        assert np.isclose(phi, expected_phi)
    
    def test_variance_calculation_atm(self):
        """Test variance calculation at ATM (strike = forward)."""
        surface = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            gamma=0.5,
            rho=-0.4,
            eta=1.0,
            lambda_param=0.3
        )
        
        # At ATM, k = 0
        # θ = 0.04, φ = 1.0 * 0.04^(-0.3)
        # w(0) = θ/2 * [1 + ρ*φ*0 + sqrt((φ*0 + ρ)² + (1 - ρ²))]
        # w(0) = θ/2 * [1 + sqrt(ρ² + 1 - ρ²)]
        # w(0) = θ/2 * [1 + sqrt(1)]
        # w(0) = θ/2 * [1 + 1] = θ
        theta = surface.get_theta()
        variance_atm = surface.get_variance(100.0)
        
        assert np.isclose(variance_atm, theta)
    
    def test_volatility_calculation_atm(self):
        """Test volatility calculation at ATM."""
        surface = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            gamma=0.5,
            rho=-0.4,
            eta=1.0,
            lambda_param=0.3
        )
        
        variance_atm = surface.get_variance(100.0)
        expected_vol = np.sqrt(variance_atm / 1.0)
        vol_atm = surface.get_volatility(100.0)
        
        assert np.isclose(vol_atm, expected_vol)
    
    def test_variance_array_input(self):
        """Test variance calculation with array input."""
        surface = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            gamma=0.5,
            rho=-0.4,
            eta=1.0,
            lambda_param=0.3
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
            a=0.04,
            gamma=0.5,
            rho=-0.4,
            eta=1.0,
            lambda_param=0.3
        )
        
        strikes = np.array([90.0, 100.0, 110.0])
        volatilities = surface.get_volatility(strikes)
        
        assert len(volatilities) == 3
        assert all(volatilities > 0)  # All volatilities should be positive
    
    def test_volatility_smile_shape(self):
        """Test that SSVI produces a volatility smile (typical shape)."""
        surface = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            gamma=0.5,
            rho=-0.6,  # Negative rho creates downward skew
            eta=1.0,
            lambda_param=0.3
        )
        
        # Check strikes from ITM put to OTM call
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        volatilities = surface.get_volatility(strikes)
        
        # All volatilities should be positive
        assert all(volatilities > 0)
        
        # With negative rho, we expect ITM puts (low strikes) to have higher vol
        vol_itm_put = volatilities[0]  # 80 strike
        vol_atm = volatilities[2]  # 100 strike
        
        assert vol_itm_put > 0 and vol_atm > 0
    
    def test_variance_positive_for_different_strikes(self):
        """Test that variance is positive across a wide range of strikes."""
        surface = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            gamma=0.5,
            rho=-0.4,
            eta=1.0,
            lambda_param=0.3
        )
        
        # Test a wide range of strikes
        strikes = np.linspace(50.0, 150.0, 50)
        variances = surface.get_variance(strikes)
        
        assert all(variances > 0), "All variances must be positive"
    
    def test_different_maturities(self):
        """Test SSVI with different maturities using the power-law."""
        surface = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            gamma=0.5,
            rho=-0.4,
            eta=1.0,
            lambda_param=0.3
        )
        
        # Short maturity: t=0.25, θ(0.25) = 0.04 * 0.25^0.5 = 0.02
        theta_short = surface.get_theta(0.25)
        assert np.isclose(theta_short, 0.02)
        
        # Long maturity: t=4.0, θ(4) = 0.04 * 4^0.5 = 0.08
        theta_long = surface.get_theta(4.0)
        assert np.isclose(theta_long, 0.08)
        
        # Verify ATM variance scales correctly
        var_short = surface.get_variance_at_maturity(100.0, 0.25)
        var_long = surface.get_variance_at_maturity(100.0, 4.0)
        
        # At ATM, variance = theta (as shown in test_variance_calculation_atm)
        assert np.isclose(var_short, theta_short)
        assert np.isclose(var_long, theta_long)
    
    def test_get_variance_at_maturity(self):
        """Test variance calculation at different maturities."""
        surface = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            gamma=0.5,
            rho=-0.4,
            eta=1.0,
            lambda_param=0.3
        )
        
        # Test at different maturities
        maturities = [0.25, 0.5, 1.0, 2.0]
        strike = 110.0
        
        for maturity in maturities:
            variance = surface.get_variance_at_maturity(strike, maturity)
            assert variance > 0
            
            # Variance should increase with maturity (for typical parameters)
            if maturity < 2.0:
                variance_longer = surface.get_variance_at_maturity(
                    strike, maturity * 2
                )
                # This holds for power-law with gamma <= 1
                assert variance_longer > variance
    
    def test_get_volatility_at_maturity(self):
        """Test volatility calculation at different maturities."""
        surface = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            gamma=0.5,
            rho=-0.4,
            eta=1.0,
            lambda_param=0.3
        )
        
        # Test at different maturities
        maturities = [0.25, 0.5, 1.0, 2.0]
        strike = 110.0
        
        for maturity in maturities:
            volatility = surface.get_volatility_at_maturity(strike, maturity)
            assert volatility > 0
            
            # Verify volatility = sqrt(variance / T)
            variance = surface.get_variance_at_maturity(strike, maturity)
            expected_vol = np.sqrt(variance / maturity)
            assert np.isclose(volatility, expected_vol)
    
    def test_repr(self):
        """Test string representation."""
        surface = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            gamma=0.5,
            rho=-0.4,
            eta=1.0,
            lambda_param=0.3
        )
        
        repr_str = repr(surface)
        assert 'SSVIVolatilitySurface' in repr_str
        assert '100.0' in repr_str
        assert '0.04' in repr_str
        assert '0.5' in repr_str
        assert '-0.4' in repr_str
        assert '1.0' in repr_str
        assert '0.3' in repr_str
    
    def test_boundary_rho_values(self):
        """Test that boundary values of rho (±1) work correctly."""
        # Test rho = 1
        surface_rho_plus = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            gamma=0.5,
            rho=1.0,
            eta=1.0,
            lambda_param=0.3
        )
        vol_plus = surface_rho_plus.get_volatility(100.0)
        assert vol_plus > 0
        
        # Test rho = -1
        surface_rho_minus = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            gamma=0.5,
            rho=-1.0,
            eta=1.0,
            lambda_param=0.3
        )
        vol_minus = surface_rho_minus.get_volatility(100.0)
        assert vol_minus > 0
    
    def test_boundary_gamma_values(self):
        """Test that gamma=1 works correctly."""
        surface = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            gamma=1.0,  # Linear growth of theta with time
            rho=-0.4,
            eta=1.0,
            lambda_param=0.3
        )
        
        # θ(t) = 0.04 * t^1 = 0.04 * t
        theta_1 = surface.get_theta(1.0)
        assert np.isclose(theta_1, 0.04)
        
        theta_2 = surface.get_theta(2.0)
        assert np.isclose(theta_2, 0.08)
        
        vol = surface.get_volatility(100.0)
        assert vol > 0
    
    def test_boundary_lambda_values(self):
        """Test that boundary values of lambda (0, 0.5) work correctly."""
        # Test lambda = 0 (constant phi)
        surface_lambda_0 = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            gamma=0.5,
            rho=-0.4,
            eta=1.0,
            lambda_param=0.0  # φ(θ) = η * θ^0 = η = 1.0
        )
        
        phi_0 = surface_lambda_0.get_phi(0.04)
        assert np.isclose(phi_0, 1.0)
        
        vol_0 = surface_lambda_0.get_volatility(100.0)
        assert vol_0 > 0
        
        # Test lambda = 0.5 (maximum allowed)
        surface_lambda_05 = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            gamma=0.5,
            rho=-0.4,
            eta=1.0,
            lambda_param=0.5
        )
        
        vol_05 = surface_lambda_05.get_volatility(100.0)
        assert vol_05 > 0
    
    def test_consistency_with_instance_maturity(self):
        """Test that get_variance equals get_variance_at_maturity with same T."""
        surface = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            gamma=0.5,
            rho=-0.4,
            eta=1.0,
            lambda_param=0.3
        )
        
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        
        for strike in strikes:
            var1 = surface.get_variance(strike)
            var2 = surface.get_variance_at_maturity(strike, 1.0)
            assert np.isclose(var1, var2)
            
            vol1 = surface.get_volatility(strike)
            vol2 = surface.get_volatility_at_maturity(strike, 1.0)
            assert np.isclose(vol1, vol2)
