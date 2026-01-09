"""Analytical calculator for closed-form option pricing formulas."""

import numpy as np
from typing import Dict, Optional

from Analytics.distributions import normal_cdf, normal_pdf
from Models.black import BlackModel
from Securities.european_option import EuropeanOption


class AnalyticalCalculator:
    """
    Analytical calculator for pricing European options using closed-form formulas.
    
    This calculator implements the Black-Scholes/Black-76 formula for European options.
    """
    
    def price(
        self,
        model: BlackModel,
        security: EuropeanOption
    ) -> float:
        """
        Calculate the price of a European option using the Black formula.
        
        The Black formula is:
        - Call: D * [F * N(d1) - K * N(d2)]
        - Put: D * [K * N(-d2) - F * N(-d1)]
        
        where:
        - d1 = [ln(F/K) + 0.5 * σ² * T] / (σ * √T)
        - d2 = d1 - σ * √T
        - D is the discount factor
        - F is the forward price
        - K is the strike
        - σ is the volatility
        - T is the time to maturity
        
        Args:
            model: BlackModel containing forward, volatility, and discount factor.
            security: EuropeanOption containing strike, maturity, and option type.
            
        Returns:
            The option price.
            
        Examples:
            >>> model = BlackModel(forward=100.0, volatility=0.2, discount_factor=1.0)
            >>> call = EuropeanOption(strike=100.0, maturity=1.0, option_type='call')
            >>> calculator = AnalyticalCalculator()
            >>> price = calculator.price(model, call)
            >>> 7.5 < price < 8.5  # Approximate ATM call price
            True
        """
        F = model.forward
        K = security.strike
        T = security.maturity
        sigma = model.volatility
        D = model.discount_factor
        
        # Handle edge case of zero volatility
        if sigma == 0:
            if security.option_type == 'call':
                return D * max(F - K, 0.0)
            else:
                return D * max(K - F, 0.0)
        
        # Calculate d1 and d2
        sqrt_T = np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        # Calculate option price
        if security.option_type == 'call':
            price = D * (F * normal_cdf(d1) - K * normal_cdf(d2))
        else:  # put
            price = D * (K * normal_cdf(-d2) - F * normal_cdf(-d1))
        
        return float(price)
    
    def greeks(
        self,
        model: BlackModel,
        security: EuropeanOption
    ) -> Dict[str, float]:
        """
        Calculate option Greeks using analytical formulas.
        
        Greeks calculated:
        - delta: ∂V/∂F (sensitivity to forward price)
        - gamma: ∂²V/∂F² (convexity)
        - vega: ∂V/∂σ (sensitivity to volatility)
        - theta: ∂V/∂T (time decay, per year)
        
        Args:
            model: BlackModel containing forward, volatility, and discount factor.
            security: EuropeanOption containing strike, maturity, and option type.
            
        Returns:
            Dictionary containing delta, gamma, vega, and theta.
            
        Examples:
            >>> model = BlackModel(forward=100.0, volatility=0.2, discount_factor=1.0)
            >>> call = EuropeanOption(strike=100.0, maturity=1.0, option_type='call')
            >>> calculator = AnalyticalCalculator()
            >>> greeks = calculator.greeks(model, call)
            >>> 0.4 < greeks['delta'] < 0.6  # ATM call delta around 0.5
            True
            >>> greeks['gamma'] > 0  # Always positive for long options
            True
        """
        F = model.forward
        K = security.strike
        T = security.maturity
        sigma = model.volatility
        D = model.discount_factor
        
        # Handle edge case of zero volatility
        if sigma == 0:
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'vega': 0.0,
                'theta': 0.0
            }
        
        # Calculate d1 and d2
        sqrt_T = np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        # Common terms
        phi_d1 = normal_pdf(d1)
        N_d1 = normal_cdf(d1)
        N_d2 = normal_cdf(d2)
        
        # Delta
        if security.option_type == 'call':
            delta = D * N_d1
        else:  # put
            delta = -D * normal_cdf(-d1)
        
        # Gamma (same for call and put)
        gamma = D * phi_d1 / (F * sigma * sqrt_T)
        
        # Vega (same for call and put, per 1% volatility change)
        vega = D * F * phi_d1 * sqrt_T / 100.0
        
        # Theta (per year)
        if security.option_type == 'call':
            theta = -D * F * phi_d1 * sigma / (2 * sqrt_T)
        else:  # put
            theta = -D * F * phi_d1 * sigma / (2 * sqrt_T)
        
        return {
            'delta': float(delta),
            'gamma': float(gamma),
            'vega': float(vega),
            'theta': float(theta)
        }
