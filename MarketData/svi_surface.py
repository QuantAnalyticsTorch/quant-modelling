"""SVI (Stochastic Volatility Inspired) volatility surface parametrization.

The SVI model provides a parametric form for the implied volatility surface
that is commonly used in practice due to its flexibility and simplicity.

Reference:
    Gatheral, J. (2004). "A parsimonious arbitrage-free implied volatility parameterization 
    with application to the valuation of volatility derivatives."
"""

from typing import Union
import numpy as np
from .volatility_surface import VolatilitySurface


class SVIVolatilitySurface(VolatilitySurface):
    """
    SVI (Stochastic Volatility Inspired) volatility surface.
    
    The raw SVI parametrization models total implied variance as:
        w(k) = a + b * [ρ * (k - m) + sqrt((k - m)² + σ²)]
    
    where:
        - k = log(K/F) is the log-moneyness
        - w(k) is the total implied variance
        - a, b, ρ (rho), m, σ (sigma) are the SVI parameters
    
    The implied volatility is then: vol(k) = sqrt(w(k) / T)
    
    Attributes:
        forward: Forward price of the underlying asset.
        time_to_maturity: Time to maturity in years.
        a: Vertical shift parameter (controls ATM variance level).
        b: Slope parameter (controls variance sensitivity to moneyness).
        rho: Correlation parameter (controls skew, must be in [-1, 1]).
        m: Horizontal shift parameter (controls ATM location).
        sigma: Curvature parameter (controls smile shape, must be positive).
    
    Examples:
        >>> surface = SVIVolatilitySurface(
        ...     forward=100.0,
        ...     time_to_maturity=1.0,
        ...     a=0.04,
        ...     b=0.1,
        ...     rho=-0.4,
        ...     m=0.0,
        ...     sigma=0.2
        ... )
        >>> vol_atm = surface.get_volatility(100.0)
        >>> vol_otm = surface.get_volatility(120.0)
    """
    
    def __init__(
        self,
        forward: float,
        time_to_maturity: float,
        a: float,
        b: float,
        rho: float,
        m: float,
        sigma: float
    ):
        """
        Initialize the SVI volatility surface.
        
        Args:
            forward: Forward price of the underlying asset.
            time_to_maturity: Time to maturity in years.
            a: Vertical shift parameter.
            b: Slope parameter (must be non-negative).
            rho: Correlation parameter (must be in [-1, 1]).
            m: Horizontal shift parameter.
            sigma: Curvature parameter (must be positive).
            
        Raises:
            ValueError: If parameters violate constraints.
        """
        super().__init__(forward, time_to_maturity)
        
        # Validate SVI parameters
        if b < 0:
            raise ValueError("Parameter b must be non-negative")
        if not -1 <= rho <= 1:
            raise ValueError("Parameter rho must be in [-1, 1]")
        if sigma <= 0:
            raise ValueError("Parameter sigma must be positive")
        
        # Additional constraint to ensure non-negative variance
        # w(k) >= 0 requires: a + b * sigma * sqrt(1 - rho^2) >= 0
        min_variance = a + b * sigma * np.sqrt(1 - rho**2)
        if min_variance < 0:
            raise ValueError(
                f"Parameters lead to negative variance. "
                f"Constraint violated: a + b * sigma * sqrt(1 - rho^2) = {min_variance:.6f} < 0"
            )
        
        self.a = a
        self.b = b
        self.rho = rho
        self.m = m
        self.sigma = sigma
    
    def get_variance(self, strike: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Get the total implied variance at the given strike(s).
        
        The raw SVI formula:
            w(k) = a + b * [ρ * (k - m) + sqrt((k - m)² + σ²)]
        
        Args:
            strike: Strike price(s) at which to compute variance.
            
        Returns:
            Total implied variance at the given strike(s).
        """
        k = self.get_log_moneyness(strike)
        k_shifted = k - self.m
        
        # Compute the SVI variance formula
        variance = self.a + self.b * (
            self.rho * k_shifted + np.sqrt(k_shifted**2 + self.sigma**2)
        )
        
        return variance
    
    def get_volatility(self, strike: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Get the implied volatility at the given strike(s).
        
        Volatility is computed as: vol(k) = sqrt(w(k) / T)
        
        Args:
            strike: Strike price(s) at which to compute volatility.
            
        Returns:
            Implied volatility (annualized) at the given strike(s).
        """
        variance = self.get_variance(strike)
        volatility = np.sqrt(variance / self.time_to_maturity)
        
        return volatility
    
    def __repr__(self) -> str:
        """String representation of the SVI surface."""
        return (
            f"SVIVolatilitySurface(forward={self.forward}, "
            f"time_to_maturity={self.time_to_maturity}, "
            f"a={self.a}, b={self.b}, rho={self.rho}, m={self.m}, sigma={self.sigma})"
        )
