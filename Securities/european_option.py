"""European option security definition."""

from typing import Literal


class EuropeanOption:
    """
    European option payoff definition.
    
    A European option can only be exercised at expiration.
    
    Attributes:
        strike: Strike price of the option.
        maturity: Time to maturity in years.
        option_type: Type of option, either 'call' or 'put'.
    """
    
    def __init__(
        self,
        strike: float,
        maturity: float,
        option_type: Literal['call', 'put']
    ):
        """
        Initialize a European option.
        
        Args:
            strike: Strike price (K).
            maturity: Time to maturity in years (T).
            option_type: Type of option, either 'call' or 'put'.
            
        Examples:
            >>> option = EuropeanOption(strike=100.0, maturity=1.0, option_type='call')
            >>> option.strike
            100.0
            >>> option.maturity
            1.0
            >>> option.option_type
            'call'
        """
        if strike <= 0:
            raise ValueError("Strike must be positive")
        if maturity <= 0:
            raise ValueError("Maturity must be positive")
        if option_type not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")
            
        self.strike = strike
        self.maturity = maturity
        self.option_type = option_type
    
    def payoff(self, spot: float) -> float:
        """
        Calculate the payoff at maturity.
        
        Args:
            spot: Spot price of the underlying at maturity.
            
        Returns:
            The option payoff.
            
        Examples:
            >>> call = EuropeanOption(strike=100.0, maturity=1.0, option_type='call')
            >>> call.payoff(110.0)
            10.0
            >>> call.payoff(90.0)
            0.0
            >>> put = EuropeanOption(strike=100.0, maturity=1.0, option_type='put')
            >>> put.payoff(90.0)
            10.0
            >>> put.payoff(110.0)
            0.0
        """
        if self.option_type == 'call':
            return max(spot - self.strike, 0.0)
        else:  # put
            return max(self.strike - spot, 0.0)
    
    def __repr__(self) -> str:
        """String representation of the option."""
        return (f"EuropeanOption(strike={self.strike}, "
                f"maturity={self.maturity}, "
                f"option_type='{self.option_type}')")
