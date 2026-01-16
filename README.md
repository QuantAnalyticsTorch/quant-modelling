# quant-modelling

A quantitative modeling library for pricing financial derivatives using the Black-Scholes model and related methods.

## Features

- **Black Model**: Implementation of the Black-76 model for pricing European options on forwards
- **SVI Volatility Surface**: Stochastic Volatility Inspired (SVI) parametrization for modeling volatility smiles and skews
- **SSVI Volatility Surface**: Surface SVI (SSVI) arbitrage-free parametrization for modeling volatility surfaces across maturities
- **Analytical Calculator**: Closed-form pricing formulas for European options
- **Greek Calculations**: Delta, Gamma, Vega, and Theta for risk management
- **European Options**: Support for call and put options
- **Type-Safe**: Extensive use of Python type hints
- **Well-Tested**: Comprehensive test suite with 75+ tests

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

### Using SVI Volatility Surfaces

```python
from MarketData.svi_surface import SVIVolatilitySurface
import numpy as np

# Create an SVI volatility surface
surface = SVIVolatilitySurface(
    forward=100.0,           # Forward price
    time_to_maturity=1.0,    # 1 year to maturity
    a=0.04,                  # Vertical shift (ATM variance level)
    b=0.15,                  # Slope parameter
    rho=-0.6,                # Correlation (negative = downward skew)
    m=0.0,                   # Horizontal shift
    sigma=0.3                # Curvature parameter
)

# Get implied volatility at a specific strike
vol_atm = surface.get_volatility(100.0)
print(f"ATM Volatility: {vol_atm:.4f}")

# Get volatilities across multiple strikes (vectorized)
strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
volatilities = surface.get_volatility(strikes)

# Get total variance
variance = surface.get_variance(100.0)
```

See `example_svi_surface.py` for a complete demonstration of SVI functionality.

### Using SSVI Volatility Surfaces

```python
from MarketData.ssvi_surface import SSVIVolatilitySurface
import numpy as np

# Create an SSVI volatility surface
surface = SSVIVolatilitySurface(
    forward=100.0,           # Forward price
    time_to_maturity=1.0,    # 1 year to maturity
    theta=0.04,              # ATM total variance
    rho=-0.3,                # Correlation (negative = downward skew)
    gamma=0.5,               # Power-law exponent
    eta=1.5                  # Scale parameter
)

# Get implied volatility at a specific strike
vol_atm = surface.get_volatility(100.0)
print(f"ATM Volatility: {vol_atm:.4f}")

# Get volatilities across multiple strikes (vectorized)
strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
volatilities = surface.get_volatility(strikes)

# Get total variance
variance = surface.get_variance(100.0)
```

See `example_ssvi_surface.py` for a complete demonstration of SSVI functionality.

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

## Examples

See `example_black_scholes.py` for pricing European options with the Black model:

```bash
python example_black_scholes.py
```

See `example_svi_surface.py` for working with SVI volatility surfaces:

```bash
python example_svi_surface.py
```

See `example_ssvi_surface.py` for working with SSVI volatility surfaces:

```bash
python example_ssvi_surface.py
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

### SVI Volatility Surface

The SVI (Stochastic Volatility Inspired) model provides a parametric form for the volatility surface. The raw SVI formula for total implied variance is:

**Variance Formula:**
```
w(k) = a + b × [ρ × (k - m) + √((k - m)² + σ²)]
```

Where:
- `k = ln(K/F)` = Log-moneyness
- `w(k)` = Total implied variance
- `a` = Vertical shift parameter (controls ATM variance level)
- `b` = Slope parameter (must be ≥ 0)
- `ρ` = Correlation parameter (must be in [-1, 1], controls skew)
- `m` = Horizontal shift parameter
- `σ` = Curvature parameter (must be > 0, controls smile shape)

**Implied Volatility:**
```
vol(k) = √(w(k) / T)
```

**Key Properties:**
- Negative `ρ` creates downward skew (typical for equity markets)
- Positive `ρ` creates upward skew
- Parameter `b` controls the sensitivity of variance to moneyness
- Non-negativity constraint: `a + b × σ × √(1 - ρ²) ≥ 0`

### SSVI Volatility Surface

The SSVI (Surface SVI) model extends SVI to provide an arbitrage-free parametrization across multiple maturities. The SSVI formula for total implied variance is:

**Variance Formula:**
```
w(k, θ) = θ/2 × {1 + ρ × φ(θ) × k + √[(φ(θ) × k + ρ)² + (1 - ρ²)]}
```

Where:
- `k = ln(K/F)` = Log-moneyness
- `w(k, θ)` = Total implied variance
- `θ(T)` = ATM total variance at time T (must be > 0)
- `φ(θ) = η / θ^γ` = Power-law function
- `ρ` = Correlation parameter (must be in [-1, 1], controls skew)
- `γ` = Power-law exponent (typically in [0, 1])
- `η` = Scale parameter (must be > 0)

**Implied Volatility:**
```
vol(k, θ) = √(w(k, θ) / T)
```

**Key Properties:**
- Arbitrage-free parametrization across maturities
- ATM variance equals `θ`: `w(0, θ) = θ`
- Negative `ρ` creates downward skew (typical for equity markets)
- `γ` and `η` control the wings behavior
- Arbitrage-free constraint: `|ρ| ≤ (1 - γ) / (1 + γ)` for `γ > 0`
- For `γ = 0`, the constraint simplifies and any `ρ ∈ [-1, 1]` is allowed


## Contributing

This is an actively developed project. Contributions are welcome!

## License

MIT License