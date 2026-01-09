"""Base class for volatility surfaces."""

from abc import ABC, abstractmethod
from typing import Union
import numpy as np


class VolatilitySurface(ABC):
    """
    Abstract base class for volatility surfaces.
    
    A volatility surface represents the implied volatility of options
    as a function of strike and time to maturity.
    """
    
    def __init__(self, forward: float, time_to_maturity: float):
        """
        Initialize the volatility surface.
        
        Args:
            forward: Forward price of the underlying asset.
            time_to_maturity: Time to maturity in years.
            
        Raises:
            ValueError: If forward or time_to_maturity are not positive.
        """
        if forward <= 0:
            raise ValueError("Forward price must be positive")
        if time_to_maturity <= 0:
            raise ValueError("Time to maturity must be positive")
            
        self.forward = forward
        self.time_to_maturity = time_to_maturity
    
    @abstractmethod
    def get_volatility(self, strike: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Get the implied volatility at the given strike(s).
        
        Args:
            strike: Strike price(s) at which to compute volatility.
            
        Returns:
            Implied volatility (annualized) at the given strike(s).
        """
        pass
    
    @abstractmethod
    def get_variance(self, strike: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Get the total implied variance at the given strike(s).
        
        Args:
            strike: Strike price(s) at which to compute variance.
            
        Returns:
            Total implied variance at the given strike(s).
        """
        pass
    
    def get_log_moneyness(self, strike: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute log-moneyness: log(K/F).
        
        Args:
            strike: Strike price(s).
            
        Returns:
            Log-moneyness at the given strike(s).
        """
        return np.log(strike / self.forward)
