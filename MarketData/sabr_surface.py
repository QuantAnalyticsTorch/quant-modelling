"""SABR (Stochastic Alpha, Beta, Rho) volatility surface parametrization.

The SABR model is a stochastic volatility model widely used in interest rate
and foreign exchange markets for modeling the volatility smile.

Reference:
    Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002).
    "Managing smile risk." Wilmott Magazine, September, 84-108.
"""

from typing import Union
import numpy as np
from .volatility_surface import VolatilitySurface


class SABRVolatilitySurface(VolatilitySurface):
    """
    SABR (Stochastic Alpha, Beta, Rho) volatility surface.
    
    The SABR model describes the forward price F and its volatility α through
    the stochastic differential equations:
        dF_t = α_t F_t^β dW_t^F
        dα_t = ν α_t dW_t^α
        dW_t^F dW_t^α = ρ dt
    
    The implied volatility is approximated using Hagan's formula for the case
    where the strike K is close to the forward F.
    
    Parameters:
        alpha: Initial volatility level (must be positive).
        beta: CEV exponent parameter (must be in [0, 1]).
              β = 0: Normal model
              β = 0.5: CIR (square-root) model
              β = 1: Lognormal model
        rho: Correlation between forward and volatility (must be in [-1, 1]).
        nu: Volatility of volatility (must be non-negative).
    
    Attributes:
        forward: Forward price of the underlying asset.
        time_to_maturity: Time to maturity in years.
        alpha: Initial volatility level.
        beta: CEV exponent parameter.
        rho: Correlation parameter.
        nu: Volatility of volatility.
    
    Examples:
        >>> # Lognormal SABR (β = 1) - typical for FX markets
        >>> surface = SABRVolatilitySurface(
        ...     forward=100.0,
        ...     time_to_maturity=1.0,
        ...     alpha=0.25,
        ...     beta=1.0,
        ...     rho=-0.3,
        ...     nu=0.4
        ... )
        >>> vol_atm = surface.get_volatility(100.0)
        >>> 
        >>> # Normal SABR (β = 0) - typical for interest rates
        >>> surface_normal = SABRVolatilitySurface(
        ...     forward=0.05,
        ...     time_to_maturity=5.0,
        ...     alpha=0.01,
        ...     beta=0.0,
        ...     rho=0.0,
        ...     nu=0.2
        ... )
        >>> vol = surface_normal.get_volatility(0.05)
    """
    
    def __init__(
        self,
        forward: float,
        time_to_maturity: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float
    ):
        """
        Initialize the SABR volatility surface.
        
        Args:
            forward: Forward price of the underlying asset.
            time_to_maturity: Time to maturity in years.
            alpha: Initial volatility level (must be positive).
            beta: CEV exponent (must be in [0, 1]).
            rho: Correlation parameter (must be in [-1, 1]).
            nu: Volatility of volatility (must be non-negative).
            
        Raises:
            ValueError: If parameters violate constraints.
        """
        super().__init__(forward, time_to_maturity)
        
        # Validate SABR parameters
        if alpha <= 0:
            raise ValueError("Parameter alpha must be positive")
        if not 0 <= beta <= 1:
            raise ValueError("Parameter beta must be in [0, 1]")
        if not -1 <= rho <= 1:
            raise ValueError("Parameter rho must be in [-1, 1]")
        if nu < 0:
            raise ValueError("Parameter nu must be non-negative")
        
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
    
    def get_volatility(self, strike: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Get the implied volatility at the given strike(s) using Hagan's approximation.
        
        The approximation is most accurate when the strike is close to the forward
        and for moderate values of the volatility of volatility (nu).
        
        Args:
            strike: Strike price(s) at which to compute volatility.
            
        Returns:
            Implied volatility (annualized) at the given strike(s).
        """
        K = np.asarray(strike)
        F = self.forward
        T = self.time_to_maturity
        
        # Handle ATM case separately for numerical stability
        is_atm = np.isclose(K, F, rtol=1e-10, atol=1e-10)
        
        if np.isscalar(strike):
            if is_atm:
                return self._get_atm_volatility()
            else:
                return self._get_volatility_hagan(strike)
        else:
            # Vectorized computation
            vols = np.zeros_like(K, dtype=float)
            
            # ATM strikes
            if np.any(is_atm):
                vols[is_atm] = self._get_atm_volatility()
            
            # Non-ATM strikes
            if np.any(~is_atm):
                non_atm_strikes = K[~is_atm]
                vols[~is_atm] = np.array([self._get_volatility_hagan(k) for k in non_atm_strikes])
            
            return vols
    
    def _get_atm_volatility(self) -> float:
        """
        Calculate the ATM (at-the-money) implied volatility.
        
        At K = F, Hagan's formula simplifies considerably.
        
        Returns:
            ATM implied volatility.
        """
        F = self.forward
        T = self.time_to_maturity
        alpha = self.alpha
        beta = self.beta
        rho = self.rho
        nu = self.nu
        
        # For β ≠ 1, we need F^(1-β)
        F_beta = F**(1 - beta) if beta != 1.0 else 1.0
        
        # First term: α / F^(1-β)
        if beta == 1.0:
            vol_atm = alpha
        else:
            vol_atm = alpha / F_beta
        
        # Second term: correction for stochastic volatility
        # (1 + [(2-3ρ²)ν²/24] T)
        if nu > 0:
            correction = 1.0 + (
                (2 - 3 * rho**2) * nu**2 / 24
            ) * T
            vol_atm *= correction
        
        return vol_atm
    
    def _get_volatility_hagan(self, strike: float) -> float:
        """
        Calculate implied volatility using Hagan's approximation formula.
        
        This is the full Hagan formula for strikes away from ATM.
        
        Args:
            strike: Strike price.
            
        Returns:
            Implied volatility.
        """
        K = strike
        F = self.forward
        T = self.time_to_maturity
        alpha = self.alpha
        beta = self.beta
        rho = self.rho
        nu = self.nu
        
        # Avoid division by zero and negative values
        if K <= 0 or F <= 0:
            return 0.0
        
        # For numerical stability, handle very small strikes
        if K < 1e-10 * F:
            return 0.0
        
        # Calculate FK = (F * K)^((1-β)/2)
        FK = (F * K)**((1 - beta) / 2)
        
        # Calculate log-moneyness
        log_FK = np.log(F / K)
        
        # Calculate z parameter
        if nu > 1e-10 and abs(log_FK) > 1e-10:
            if beta == 1.0:
                # For lognormal: z = (ν/α) × log(F/K)
                z = (nu / alpha) * log_FK
            else:
                # General case: z = (ν/α) × (F^(1-β) - K^(1-β)) / (1-β)
                z = (nu / alpha) * (F**(1 - beta) - K**(1 - beta)) / (1 - beta)
        else:
            z = 0.0
        
        # Calculate χ(z) - this handles the vol-of-vol adjustment
        if abs(z) < 1e-7:
            # Taylor expansion for small z to avoid numerical issues
            chi_z = 1.0
        else:
            # chi(z) = z / log[(√(1 - 2ρz + z²) + z - ρ) / (1 - ρ)]
            discriminant = 1 - 2 * rho * z + z**2
            
            # Ensure discriminant is non-negative
            if discriminant < 0:
                discriminant = 0.0
            
            sqrt_term = np.sqrt(discriminant)
            numerator_chi = sqrt_term + z - rho
            denominator_chi = 1 - rho
            
            # Handle special case when rho is close to 1
            if abs(denominator_chi) < 1e-10:
                chi_z = 1.0
            elif numerator_chi <= 0:
                chi_z = 1.0
            else:
                log_term = np.log(numerator_chi / denominator_chi)
                if abs(log_term) < 1e-10:
                    chi_z = 1.0
                else:
                    chi_z = z / log_term
        
        # Calculate the main volatility term (numerator in the SABR formula)
        if beta == 1.0:
            # Lognormal case: σ = (α / |log(F/K)|) × χ(z) × [corrections]
            # We need absolute value to ensure positive result
            if abs(log_FK) < 1e-10:
                # Near ATM
                numerator_term = alpha
            else:
                # Use the formula structure that preserves positivity
                # The key insight: for beta=1, σ_impl = α × χ(z) × |1/log(F/K)|
                numerator_term = alpha * abs(chi_z / log_FK)
        else:
            # General case
            F_beta = F**(1 - beta)
            K_beta = K**(1 - beta)
            diff = F_beta - K_beta
            
            if abs(diff) < 1e-10:
                # Near ATM
                numerator_term = alpha / F_beta
            else:
                # σ_impl = α × χ(z) × (F^(1-β) - K^(1-β)) / [log(F/K) × (1-β)]
                # This needs to be positive, so we use absolute values carefully
                numerator_term = alpha * abs(chi_z * diff / (log_FK * (1 - beta)))
        
        # Calculate the time-dependent correction terms
        # These account for the finite maturity
        if beta == 1.0:
            term1 = 0.0
        else:
            term1 = ((1 - beta)**2 / 24) * (alpha**2 / FK**2)
        
        term2 = (rho * beta * nu * alpha) / (4 * FK)
        term3 = ((2 - 3 * rho**2) / 24) * nu**2
        
        correction = 1.0 + (term1 + term2 + term3) * T
        
        implied_vol = numerator_term * correction
        
        return max(implied_vol, 0.0)  # Ensure non-negative
    
    def get_variance(self, strike: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Get the total implied variance at the given strike(s).
        
        Variance is computed as: var = vol² × T
        
        Args:
            strike: Strike price(s) at which to compute variance.
            
        Returns:
            Total implied variance at the given strike(s).
        """
        volatility = self.get_volatility(strike)
        variance = volatility**2 * self.time_to_maturity
        
        return variance
    
    def __repr__(self) -> str:
        """String representation of the SABR surface."""
        return (
            f"SABRVolatilitySurface(forward={self.forward}, "
            f"time_to_maturity={self.time_to_maturity}, "
            f"alpha={self.alpha}, beta={self.beta}, rho={self.rho}, nu={self.nu})"
        )
