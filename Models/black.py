"""Black model for option pricing.

The Black model (Black-76) is used for pricing European options on forwards and futures.
It's a special case of the Black-Scholes model where the spot price is replaced by the
forward price.
"""

import numpy as np
from typing import Optional


class BlackModel:
    """
    Black model for European option pricing.
    
    This model assumes:
    - Constant volatility
    - Log-normal distribution of the underlying forward price
    - No dividends (incorporated in forward price)
    
    Attributes:
        forward: Forward price of the underlying asset.
        volatility: Implied volatility (annualized).
        discount_factor: Discount factor to present value (e.g., exp(-r*T)).
    """
    
    def __init__(
        self,
        forward: float,
        volatility: float,
        discount_factor: float = 1.0
    ):
        """
        Initialize the Black model.
        
        Args:
            forward: Forward price of the underlying asset.
            volatility: Implied volatility (annualized, e.g., 0.2 for 20%).
            discount_factor: Discount factor for present value calculation.
                           Default is 1.0 (no discounting).
                           
        Examples:
            >>> model = BlackModel(forward=100.0, volatility=0.2, discount_factor=0.95)
            >>> model.forward
            100.0
            >>> model.volatility
            0.2
        """
        if forward <= 0:
            raise ValueError("Forward price must be positive")
        if volatility < 0:
            raise ValueError("Volatility must be non-negative")
        if discount_factor <= 0:
            raise ValueError("Discount factor must be positive")
            
        self.forward = forward
        self.volatility = volatility
        self.discount_factor = discount_factor
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (f"BlackModel(forward={self.forward}, "
                f"volatility={self.volatility}, "
                f"discount_factor={self.discount_factor})")
