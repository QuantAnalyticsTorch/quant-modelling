# quant-modelling

A quantitative modeling library for pricing financial derivatives using the Black-Scholes model and related methods.

## Features

- **Black Model**: Implementation of the Black-76 model for pricing European options on forwards
- **Analytical Calculator**: Closed-form pricing formulas for European options
- **Greek Calculations**: Delta, Gamma, Vega, and Theta for risk management
- **European Options**: Support for call and put options
- **Type-Safe**: Extensive use of Python type hints
- **Well-Tested**: Comprehensive test suite with 38+ tests

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

```python
from Models.black import BlackModel
from Securities.european_option import EuropeanOption
from Calculators.analytical import AnalyticalCalculator

# Create a Black model
model = BlackModel(
    forward=100.0,           # Forward price
    volatility=0.2,          # 20% annual volatility
    discount_factor=0.95     # Discount factor
)

# Create a European call option
call = EuropeanOption(
    strike=100.0,            # Strike price
    maturity=1.0,            # 1 year to maturity
    option_type='call'
)

# Price the option
calculator = AnalyticalCalculator()
price = calculator.price(model, call)
print(f"Option Price: ${price:.4f}")

# Calculate Greeks
greeks = calculator.greeks(model, call)
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")
print(f"Vega: {greeks['vega']:.4f}")
print(f"Theta: {greeks['theta']:.4f}")
```

## Architecture

The library follows a flexible triple pattern: **(Model, Security, Calculator)**

- **Models**: Quantitative models (e.g., BlackModel)
- **Securities**: Financial instruments (e.g., EuropeanOption)
- **Calculators**: Pricing methods (e.g., AnalyticalCalculator)

This allows pricing the same security under different models or using different numerical methods.

### Directory Structure

```
quant-modelling/
├── Analytics/          # Core numerical functions (distributions, etc.)
├── Models/             # Financial models (Black, etc.)
├── Securities/         # Financial instruments (options, etc.)
├── Calculators/        # Pricing methods (analytical, MC, PDE)
├── MarketData/         # Market data structures (curves, surfaces)
├── Scenarios/          # Scenario generation
├── tests/              # Test suite
└── example_black_scholes.py  # Example usage
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Example

See `example_black_scholes.py` for a complete example:

```bash
python example_black_scholes.py
```

## Mathematical Background

### Black Model (Black-76)

The Black model is used for pricing European options on forwards:

**Call Option:**
```
C = D × [F × N(d₁) - K × N(d₂)]
```

**Put Option:**
```
P = D × [K × N(-d₂) - F × N(-d₁)]
```

Where:
- `d₁ = [ln(F/K) + 0.5σ²T] / (σ√T)`
- `d₂ = d₁ - σ√T`
- `D` = Discount factor
- `F` = Forward price
- `K` = Strike price
- `σ` = Volatility
- `T` = Time to maturity
- `N(·)` = Standard normal CDF

### Put-Call Parity

The library validates put-call parity:
```
C - P = D × (F - K)
```

## Contributing

This is an actively developed project. Contributions are welcome!

## License

MIT License