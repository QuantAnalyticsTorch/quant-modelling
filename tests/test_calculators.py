"""Tests for Calculators module - Black-Scholes analytical pricer."""

import numpy as np
import pytest
from Models.black import BlackModel
from Securities.european_option import EuropeanOption
from Calculators.analytical import AnalyticalCalculator


class TestAnalyticalCalculator:
    """Test cases for AnalyticalCalculator."""
    
    def test_atm_call_price(self):
        """Test at-the-money call option pricing."""
        model = BlackModel(forward=100.0, volatility=0.2, discount_factor=1.0)
        call = EuropeanOption(strike=100.0, maturity=1.0, option_type='call')
        calculator = AnalyticalCalculator()
        
        price = calculator.price(model, call)
        # ATM call with 20% vol and 1 year should be around 7.97
        assert 7.5 < price < 8.5
    
    def test_atm_put_price(self):
        """Test at-the-money put option pricing."""
        model = BlackModel(forward=100.0, volatility=0.2, discount_factor=1.0)
        put = EuropeanOption(strike=100.0, maturity=1.0, option_type='put')
        calculator = AnalyticalCalculator()
        
        price = calculator.price(model, put)
        # ATM put should equal ATM call when forward = strike and D=1
        assert 7.5 < price < 8.5
    
    def test_put_call_parity(self):
        """Test put-call parity: C - P = D * (F - K)."""
        model = BlackModel(forward=100.0, volatility=0.2, discount_factor=0.95)
        call = EuropeanOption(strike=100.0, maturity=1.0, option_type='call')
        put = EuropeanOption(strike=100.0, maturity=1.0, option_type='put')
        calculator = AnalyticalCalculator()
        
        call_price = calculator.price(model, call)
        put_price = calculator.price(model, put)
        
        # C - P should equal D * (F - K)
        expected_diff = model.discount_factor * (model.forward - call.strike)
        actual_diff = call_price - put_price
        
        assert abs(actual_diff - expected_diff) < 1e-10
    
    def test_deep_itm_call(self):
        """Test deep in-the-money call option."""
        model = BlackModel(forward=150.0, volatility=0.2, discount_factor=1.0)
        call = EuropeanOption(strike=100.0, maturity=1.0, option_type='call')
        calculator = AnalyticalCalculator()
        
        price = calculator.price(model, call)
        # Should be close to intrinsic value (50) for deep ITM
        assert price > 45.0
        assert price < 55.0
    
    def test_deep_otm_call(self):
        """Test deep out-of-the-money call option."""
        model = BlackModel(forward=50.0, volatility=0.2, discount_factor=1.0)
        call = EuropeanOption(strike=100.0, maturity=1.0, option_type='call')
        calculator = AnalyticalCalculator()
        
        price = calculator.price(model, call)
        # Should be close to zero for deep OTM
        assert price < 1.0
        assert price >= 0.0
    
    def test_zero_volatility_call(self):
        """Test call option pricing with zero volatility."""
        model = BlackModel(forward=110.0, volatility=0.0, discount_factor=1.0)
        call = EuropeanOption(strike=100.0, maturity=1.0, option_type='call')
        calculator = AnalyticalCalculator()
        
        price = calculator.price(model, call)
        # With zero vol, price should equal discounted intrinsic value
        expected = model.discount_factor * max(model.forward - call.strike, 0.0)
        assert abs(price - expected) < 1e-10
    
    def test_zero_volatility_put(self):
        """Test put option pricing with zero volatility."""
        model = BlackModel(forward=90.0, volatility=0.0, discount_factor=1.0)
        put = EuropeanOption(strike=100.0, maturity=1.0, option_type='put')
        calculator = AnalyticalCalculator()
        
        price = calculator.price(model, put)
        # With zero vol, price should equal discounted intrinsic value
        expected = model.discount_factor * max(put.strike - model.forward, 0.0)
        assert abs(price - expected) < 1e-10
    
    def test_greeks_call(self):
        """Test Greeks calculation for call option."""
        model = BlackModel(forward=100.0, volatility=0.2, discount_factor=1.0)
        call = EuropeanOption(strike=100.0, maturity=1.0, option_type='call')
        calculator = AnalyticalCalculator()
        
        greeks = calculator.greeks(model, call)
        
        # ATM call delta should be around 0.5
        assert 0.4 < greeks['delta'] < 0.6
        
        # Gamma should be positive
        assert greeks['gamma'] > 0
        
        # Vega should be positive
        assert greeks['vega'] > 0
        
        # Theta should be negative (time decay)
        assert greeks['theta'] < 0
    
    def test_greeks_put(self):
        """Test Greeks calculation for put option."""
        model = BlackModel(forward=100.0, volatility=0.2, discount_factor=1.0)
        put = EuropeanOption(strike=100.0, maturity=1.0, option_type='put')
        calculator = AnalyticalCalculator()
        
        greeks = calculator.greeks(model, put)
        
        # ATM put delta should be around -0.5
        assert -0.6 < greeks['delta'] < -0.4
        
        # Gamma should be positive (same as call)
        assert greeks['gamma'] > 0
        
        # Vega should be positive (same as call)
        assert greeks['vega'] > 0
        
        # Theta should be negative
        assert greeks['theta'] < 0
    
    def test_greeks_zero_volatility(self):
        """Test Greeks with zero volatility."""
        model = BlackModel(forward=100.0, volatility=0.0, discount_factor=1.0)
        call = EuropeanOption(strike=100.0, maturity=1.0, option_type='call')
        calculator = AnalyticalCalculator()
        
        greeks = calculator.greeks(model, call)
        
        # All Greeks should be zero with zero volatility
        assert greeks['delta'] == 0.0
        assert greeks['gamma'] == 0.0
        assert greeks['vega'] == 0.0
        assert greeks['theta'] == 0.0
    
    def test_increasing_volatility_increases_price(self):
        """Test that higher volatility increases option price."""
        calculator = AnalyticalCalculator()
        call = EuropeanOption(strike=100.0, maturity=1.0, option_type='call')
        
        model_low_vol = BlackModel(forward=100.0, volatility=0.1, discount_factor=1.0)
        model_high_vol = BlackModel(forward=100.0, volatility=0.3, discount_factor=1.0)
        
        price_low = calculator.price(model_low_vol, call)
        price_high = calculator.price(model_high_vol, call)
        
        assert price_high > price_low
    
    def test_discount_factor_effect(self):
        """Test that discount factor affects option price."""
        calculator = AnalyticalCalculator()
        call = EuropeanOption(strike=100.0, maturity=1.0, option_type='call')
        
        model_no_discount = BlackModel(forward=100.0, volatility=0.2, discount_factor=1.0)
        model_with_discount = BlackModel(forward=100.0, volatility=0.2, discount_factor=0.95)
        
        price_no_discount = calculator.price(model_no_discount, call)
        price_with_discount = calculator.price(model_with_discount, call)
        
        # Discounted price should be lower
        assert price_with_discount < price_no_discount
        # Ratio should be approximately the discount factor
        assert abs(price_with_discount / price_no_discount - 0.95) < 0.01
