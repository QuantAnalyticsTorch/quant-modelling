"""Tests for Models module."""

import pytest
from Models.black import BlackModel


class TestBlackModel:
    """Test cases for BlackModel."""
    
    def test_initialization(self):
        """Test model initialization with valid parameters."""
        model = BlackModel(forward=100.0, volatility=0.2, discount_factor=0.95)
        assert model.forward == 100.0
        assert model.volatility == 0.2
        assert model.discount_factor == 0.95
    
    def test_default_discount_factor(self):
        """Test that default discount factor is 1.0."""
        model = BlackModel(forward=100.0, volatility=0.2)
        assert model.discount_factor == 1.0
    
    def test_invalid_forward(self):
        """Test that negative or zero forward raises ValueError."""
        with pytest.raises(ValueError, match="Forward price must be positive"):
            BlackModel(forward=0.0, volatility=0.2)
        with pytest.raises(ValueError, match="Forward price must be positive"):
            BlackModel(forward=-100.0, volatility=0.2)
    
    def test_invalid_volatility(self):
        """Test that negative volatility raises ValueError."""
        with pytest.raises(ValueError, match="Volatility must be non-negative"):
            BlackModel(forward=100.0, volatility=-0.1)
    
    def test_invalid_discount_factor(self):
        """Test that non-positive discount factor raises ValueError."""
        with pytest.raises(ValueError, match="Discount factor must be positive"):
            BlackModel(forward=100.0, volatility=0.2, discount_factor=0.0)
        with pytest.raises(ValueError, match="Discount factor must be positive"):
            BlackModel(forward=100.0, volatility=0.2, discount_factor=-0.5)
    
    def test_repr(self):
        """Test string representation."""
        model = BlackModel(forward=100.0, volatility=0.2, discount_factor=0.95)
        repr_str = repr(model)
        assert 'BlackModel' in repr_str
        assert '100.0' in repr_str
        assert '0.2' in repr_str
        assert '0.95' in repr_str
