"""SSVI (Surface SVI) volatility surface parametrization.

The SSVI model provides a parametric form for the implied volatility surface
across multiple maturities that is arbitrage-free and commonly used in practice.

Reference:
    Gatheral, J. and Jacquier, A. (2014). "Arbitrage-free SVI volatility surfaces."
    Quantitative Finance, 14(1), 59-71.
"""

from typing import Union
import numpy as np
from .volatility_surface import VolatilitySurface


class SSVIVolatilitySurface(VolatilitySurface):
    """
    SSVI (Surface SVI) volatility surface.
    
    The SSVI parametrization models total implied variance as:
        w(k, θ) = θ/2 × {1 + ρ × φ(θ) × k + sqrt[(φ(θ) × k + ρ)² + (1 - ρ²)]}
    
    where:
        - k = log(K/F) is the log-moneyness
        - θ(t) is the ATM total variance at time t
        - φ(θ) = η / θ^γ is the power-law function
        - ρ is the correlation parameter (must be in [-1, 1])
        - γ (gamma) is the power-law exponent (typically in [0, 1])
        - η (eta) is the scale parameter (must be positive)
    
    The implied volatility is then: vol(k, θ) = sqrt(w(k, θ) / T)
    
    Attributes:
        forward: Forward price of the underlying asset.
        time_to_maturity: Time to maturity in years.
        theta: ATM total variance (must be positive).
        rho: Correlation parameter (must be in [-1, 1]).
        gamma: Power-law exponent (typically in [0, 1]).
        eta: Scale parameter (must be positive).
    
    Examples:
        >>> surface = SSVIVolatilitySurface(
        ...     forward=100.0,
        ...     time_to_maturity=1.0,
        ...     theta=0.04,
        ...     rho=-0.4,
        ...     gamma=0.5,
        ...     eta=1.5
        ... )
        >>> vol_atm = surface.get_volatility(100.0)
        >>> vol_otm = surface.get_volatility(120.0)
    """
    
    def __init__(
        self,
        forward: float,
        time_to_maturity: float,
        theta: float,
        rho: float,
        gamma: float,
        eta: float
    ):
        """
        Initialize the SSVI volatility surface.
        
        Args:
            forward: Forward price of the underlying asset.
            time_to_maturity: Time to maturity in years.
            theta: ATM total variance (must be positive).
            rho: Correlation parameter (must be in [-1, 1]).
            gamma: Power-law exponent (typically in [0, 1]).
            eta: Scale parameter (must be positive).
            
        Raises:
            ValueError: If parameters violate constraints.
        """
        super().__init__(forward, time_to_maturity)
        
        # Validate SSVI parameters
        if theta <= 0:
            raise ValueError("Parameter theta must be positive")
        if not -1 <= rho <= 1:
            raise ValueError("Parameter rho must be in [-1, 1]")
        if eta <= 0:
            raise ValueError("Parameter eta must be positive")
        
        # Additional constraint for arbitrage-free surfaces
        # For gamma, typical values are in [0, 1], but we allow broader range
        # The key constraint is that φ(θ) must be well-defined
        if gamma < 0:
            raise ValueError("Parameter gamma must be non-negative")
        
        # Constraint to ensure arbitrage-free surface
        # |rho| <= (1 - gamma) / (1 + gamma) for gamma > 0
        if gamma > 0:
            max_abs_rho = (1 - gamma) / (1 + gamma)
            if abs(rho) > max_abs_rho + 1e-10:  # Small tolerance for numerical precision
                raise ValueError(
                    f"For gamma={gamma:.4f}, |rho| must be <= {max_abs_rho:.4f} "
                    f"to ensure arbitrage-free surface. Got |rho|={abs(rho):.4f}"
                )
        
        self.theta = theta
        self.rho = rho
        self.gamma = gamma
        self.eta = eta
        
        # Pre-compute constant term to optimize variance calculation
        self._rho_squared_complement = 1 - rho ** 2
    
    def _phi(self) -> float:
        """
        Compute the power-law function φ(θ) = η / θ^γ.
        
        Returns:
            The value of φ(θ).
        """
        return self.eta / (self.theta ** self.gamma)
    
    def get_variance(self, strike: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Get the total implied variance at the given strike(s).
        
        The SSVI formula:
            w(k, θ) = θ/2 × {1 + ρ × φ(θ) × k + sqrt[(φ(θ) × k + ρ)² + (1 - ρ²)]}
        
        Args:
            strike: Strike price(s) at which to compute variance.
            
        Returns:
            Total implied variance at the given strike(s).
        """
        k = self.get_log_moneyness(strike)
        phi = self._phi()
        
        # Compute the SSVI variance formula
        phi_k = phi * k
        term_inside_sqrt = (phi_k + self.rho) ** 2 + self._rho_squared_complement
        
        variance = (self.theta / 2.0) * (
            1 + self.rho * phi_k + np.sqrt(term_inside_sqrt)
        )
        
        return variance
    
    def get_volatility(self, strike: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Get the implied volatility at the given strike(s).
        
        Volatility is computed as: vol(k, θ) = sqrt(w(k, θ) / T)
        
        Args:
            strike: Strike price(s) at which to compute volatility.
            
        Returns:
            Implied volatility (annualized) at the given strike(s).
        """
        variance = self.get_variance(strike)
        volatility = np.sqrt(variance / self.time_to_maturity)
        
        return volatility
    
    def __repr__(self) -> str:
        """String representation of the SSVI surface."""
        return (
            f"SSVIVolatilitySurface(forward={self.forward}, "
            f"time_to_maturity={self.time_to_maturity}, "
            f"theta={self.theta}, rho={self.rho}, gamma={self.gamma}, eta={self.eta})"
        )
