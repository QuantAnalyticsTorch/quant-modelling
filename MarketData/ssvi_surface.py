"""SSVI (Surface SVI) volatility surface parametrization.

The SSVI model provides a parametric form for the entire implied volatility surface
across multiple maturities, ensuring no calendar spread arbitrage.

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
    
    The SSVI parametrization models total implied variance across maturities as:
        w(k, θ) = θ/2 * [1 + ρ * φ(θ) * k + sqrt((φ(θ) * k + ρ)² + (1 - ρ²))]
    
    where:
        - k = log(K/F) is the log-moneyness
        - θ = θ(t) is the ATM total variance at maturity t
        - ρ is the correlation parameter (controls skew, must be in [-1, 1])
        - φ(θ) is a function controlling the curvature of the smile
    
    The ATM variance is parameterized using a power-law:
        θ(t) = a * t^γ
    
    where:
        - a is the level parameter (must be positive)
        - γ (gamma) is the power parameter (typically in (0, 1])
    
    The curvature function φ(θ) can be:
        - Power-law: φ(θ) = η * θ^(-λ) where η > 0, λ ∈ [0, 0.5]
        - Heston-like: φ(θ) = η / (θ^λ * (1 + θ)^(1-λ)) where η > 0, λ ∈ [0, 1]
    
    This implementation uses the power-law form for φ(θ).
    
    Attributes:
        forward: Forward price of the underlying asset.
        time_to_maturity: Time to maturity in years.
        a: ATM variance level parameter (must be positive).
        gamma: ATM variance power parameter (must be in (0, 1]).
        rho: Correlation parameter (controls skew, must be in [-1, 1]).
        eta: Curvature level parameter (must be positive).
        lambda_param: Curvature power parameter (must be in [0, 0.5]).
    
    Examples:
        >>> surface = SSVIVolatilitySurface(
        ...     forward=100.0,
        ...     time_to_maturity=1.0,
        ...     a=0.04,
        ...     gamma=0.5,
        ...     rho=-0.4,
        ...     eta=1.0,
        ...     lambda_param=0.3
        ... )
        >>> vol_atm = surface.get_volatility(100.0)
        >>> vol_otm = surface.get_volatility(120.0)
    """
    
    def __init__(
        self,
        forward: float,
        time_to_maturity: float,
        a: float,
        gamma: float,
        rho: float,
        eta: float,
        lambda_param: float
    ):
        """
        Initialize the SSVI volatility surface.
        
        Args:
            forward: Forward price of the underlying asset.
            time_to_maturity: Time to maturity in years.
            a: ATM variance level parameter (must be positive).
            gamma: ATM variance power parameter (must be in (0, 1]).
            rho: Correlation parameter (must be in [-1, 1]).
            eta: Curvature level parameter (must be positive).
            lambda_param: Curvature power parameter (must be in [0, 0.5]).
            
        Raises:
            ValueError: If parameters violate constraints.
        """
        super().__init__(forward, time_to_maturity)
        
        # Validate SSVI parameters
        if a <= 0:
            raise ValueError("Parameter a must be positive")
        if gamma <= 0 or gamma > 1:
            raise ValueError("Parameter gamma must be in (0, 1]")
        if not -1 <= rho <= 1:
            raise ValueError("Parameter rho must be in [-1, 1]")
        if eta <= 0:
            raise ValueError("Parameter eta must be positive")
        if lambda_param < 0 or lambda_param > 0.5:
            raise ValueError("Parameter lambda_param must be in [0, 0.5]")
        
        self.a = a
        self.gamma = gamma
        self.rho = rho
        self.eta = eta
        self.lambda_param = lambda_param
    
    def get_theta(self, time_to_maturity: float = None) -> float:
        """
        Get ATM total variance θ(t) = a * t^γ.
        
        Args:
            time_to_maturity: Time to maturity in years. If None, uses self.time_to_maturity.
            
        Returns:
            ATM total variance at the given maturity.
        """
        if time_to_maturity is None:
            time_to_maturity = self.time_to_maturity
        
        return self.a * (time_to_maturity ** self.gamma)
    
    def get_phi(self, theta: float) -> float:
        """
        Get the curvature function φ(θ) = η * θ^(-λ).
        
        Args:
            theta: ATM total variance.
            
        Returns:
            Curvature parameter at the given theta.
        """
        return self.eta * (theta ** (-self.lambda_param))
    
    def get_variance(self, strike: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Get the total implied variance at the given strike(s).
        
        The SSVI formula:
            w(k, θ) = θ/2 * [1 + ρ * φ(θ) * k + sqrt((φ(θ) * k + ρ)² + (1 - ρ²))]
        
        Args:
            strike: Strike price(s) at which to compute variance.
            
        Returns:
            Total implied variance at the given strike(s).
        """
        k = self.get_log_moneyness(strike)
        theta = self.get_theta()
        phi = self.get_phi(theta)
        
        # SSVI variance formula
        phi_k = phi * k
        term_inside_sqrt = (phi_k + self.rho)**2 + (1 - self.rho**2)
        
        variance = (theta / 2.0) * (
            1 + self.rho * phi_k + np.sqrt(term_inside_sqrt)
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
    
    def get_variance_at_maturity(
        self, 
        strike: Union[float, np.ndarray], 
        time_to_maturity: float
    ) -> Union[float, np.ndarray]:
        """
        Get the total implied variance at a different maturity.
        
        This allows querying the surface at maturities different from the
        instance's time_to_maturity.
        
        Args:
            strike: Strike price(s) at which to compute variance.
            time_to_maturity: Time to maturity in years.
            
        Returns:
            Total implied variance at the given strike(s) and maturity.
        """
        k = np.log(strike / self.forward)
        theta = self.get_theta(time_to_maturity)
        phi = self.get_phi(theta)
        
        # SSVI variance formula
        phi_k = phi * k
        term_inside_sqrt = (phi_k + self.rho)**2 + (1 - self.rho**2)
        
        variance = (theta / 2.0) * (
            1 + self.rho * phi_k + np.sqrt(term_inside_sqrt)
        )
        
        return variance
    
    def get_volatility_at_maturity(
        self, 
        strike: Union[float, np.ndarray], 
        time_to_maturity: float
    ) -> Union[float, np.ndarray]:
        """
        Get the implied volatility at a different maturity.
        
        This allows querying the surface at maturities different from the
        instance's time_to_maturity.
        
        Args:
            strike: Strike price(s) at which to compute volatility.
            time_to_maturity: Time to maturity in years.
            
        Returns:
            Implied volatility (annualized) at the given strike(s) and maturity.
        """
        variance = self.get_variance_at_maturity(strike, time_to_maturity)
        volatility = np.sqrt(variance / time_to_maturity)
        
        return volatility
    
    def __repr__(self) -> str:
        """String representation of the SSVI surface."""
        return (
            f"SSVIVolatilitySurface(forward={self.forward}, "
            f"time_to_maturity={self.time_to_maturity}, "
            f"a={self.a}, gamma={self.gamma}, rho={self.rho}, "
            f"eta={self.eta}, lambda_param={self.lambda_param})"
        )
