"""Tests for Securities module."""

import pytest
from Securities.european_option import EuropeanOption


class TestEuropeanOption:
    """Test cases for EuropeanOption."""
    
    def test_initialization_call(self):
        """Test call option initialization."""
        option = EuropeanOption(strike=100.0, maturity=1.0, option_type='call')
        assert option.strike == 100.0
        assert option.maturity == 1.0
        assert option.option_type == 'call'
    
    def test_initialization_put(self):
        """Test put option initialization."""
        option = EuropeanOption(strike=100.0, maturity=1.0, option_type='put')
        assert option.option_type == 'put'
    
    def test_invalid_strike(self):
        """Test that invalid strike raises ValueError."""
        with pytest.raises(ValueError, match="Strike must be positive"):
            EuropeanOption(strike=0.0, maturity=1.0, option_type='call')
        with pytest.raises(ValueError, match="Strike must be positive"):
            EuropeanOption(strike=-100.0, maturity=1.0, option_type='call')
    
    def test_invalid_maturity(self):
        """Test that invalid maturity raises ValueError."""
        with pytest.raises(ValueError, match="Maturity must be positive"):
            EuropeanOption(strike=100.0, maturity=0.0, option_type='call')
        with pytest.raises(ValueError, match="Maturity must be positive"):
            EuropeanOption(strike=100.0, maturity=-1.0, option_type='call')
    
    def test_invalid_option_type(self):
        """Test that invalid option type raises ValueError."""
        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            EuropeanOption(strike=100.0, maturity=1.0, option_type='invalid')
    
    def test_call_payoff_itm(self):
        """Test call option payoff when in-the-money."""
        call = EuropeanOption(strike=100.0, maturity=1.0, option_type='call')
        assert call.payoff(110.0) == 10.0
    
    def test_call_payoff_otm(self):
        """Test call option payoff when out-of-the-money."""
        call = EuropeanOption(strike=100.0, maturity=1.0, option_type='call')
        assert call.payoff(90.0) == 0.0
    
    def test_call_payoff_atm(self):
        """Test call option payoff when at-the-money."""
        call = EuropeanOption(strike=100.0, maturity=1.0, option_type='call')
        assert call.payoff(100.0) == 0.0
    
    def test_put_payoff_itm(self):
        """Test put option payoff when in-the-money."""
        put = EuropeanOption(strike=100.0, maturity=1.0, option_type='put')
        assert put.payoff(90.0) == 10.0
    
    def test_put_payoff_otm(self):
        """Test put option payoff when out-of-the-money."""
        put = EuropeanOption(strike=100.0, maturity=1.0, option_type='put')
        assert put.payoff(110.0) == 0.0
    
    def test_put_payoff_atm(self):
        """Test put option payoff when at-the-money."""
        put = EuropeanOption(strike=100.0, maturity=1.0, option_type='put')
        assert put.payoff(100.0) == 0.0
    
    def test_repr(self):
        """Test string representation."""
        option = EuropeanOption(strike=100.0, maturity=1.0, option_type='call')
        repr_str = repr(option)
        assert 'EuropeanOption' in repr_str
        assert '100.0' in repr_str
        assert '1.0' in repr_str
        assert 'call' in repr_str
