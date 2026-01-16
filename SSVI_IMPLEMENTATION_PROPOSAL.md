# SSVI (Surface SVI) Volatility Surface Implementation Proposal

## Executive Summary

This document proposes adding SSVI (Surface Stochastic Volatility Inspired) parametrization to the quant-modelling library. While the current implementation supports SVI for modeling volatility smiles at a single maturity, SSVI extends this to provide a **consistent volatility surface across multiple maturities** while preserving no-arbitrage conditions.

## Background

### Current State: SVI
The existing `SVIVolatilitySurface` class implements the raw SVI parametrization for a single maturity slice:

```
w(k, θ) = a + b × [ρ × (k - m) + √((k - m)² + σ²)]
```

Where `θ = (a, b, ρ, m, σ)` are the five SVI parameters that must be calibrated independently for each maturity.

**Limitations:**
- Parameters must be fitted separately for each maturity
- No guarantee of calendar arbitrage-free surface across maturities
- Parameter values can change non-smoothly across time slices
- Difficult to extrapolate to maturities without market data

### SSVI Overview
SSVI (introduced by Gatheral and Jacquier, 2014) addresses these limitations by:
1. Parameterizing how SVI parameters evolve with maturity
2. Ensuring the resulting surface is free from calendar spread arbitrage
3. Providing smooth interpolation and extrapolation across maturities
4. Reducing the number of free parameters (from 5×N to ~4 total for N maturities)

## Mathematical Framework

### SSVI Parametrization

The SSVI model constrains the total implied variance to the form:

```
w(k, θ_t) = θ_t/2 × {1 + ρ × φ(θ_t) × k + √[(φ(θ_t) × k + ρ)² + (1 - ρ²)]}
```

Where:
- `k = ln(K/F)` is the log-moneyness
- `θ_t` is the ATM total variance at maturity `t`
- `ρ ∈ (-1, 1)` is the correlation parameter (typically constant across maturities)
- `φ(θ_t)` is the maturity-dependent curvature function

### Curvature Function φ(θ)

The function `φ(θ)` controls how the volatility smile evolves with maturity. Common choices include:

**1. Power Law (Recommended):**
```
φ(θ) = η / θ^γ
```
Where `η > 0` and `γ ∈ [0, 1]` are calibration parameters.
- `γ = 1/2`: Provides good fit for typical equity surfaces
- `γ = 1`: Heston-like behavior (linear in time)

**2. Heston-like:**
```
φ(θ) = η / (θ(1 + θ))
```

**3. Simple:**
```
φ(θ) = η
```
(Constant curvature, simplest case)

### ATM Variance Term Structure θ_t

The ATM variance `θ_t` must be specified for each maturity, typically through:
- Direct market calibration to ATM implied volatilities
- Parametric form: `θ_t = θ_0 × t` (linear in time for constant ATM vol)
- Interpolation/extrapolation schemes (e.g., cubic spline)

### No-Arbitrage Conditions

For the surface to be free of calendar spread arbitrage, SSVI must satisfy:

```
∂w/∂t ≥ 0  for all k, t
```

This is guaranteed when:
1. `θ_t` is non-decreasing in `t`
2. `ρ ∈ (-1, 1)` (strictly inside)
3. `φ(θ)` satisfies certain monotonicity conditions

For the power law parametrization with `γ ≤ 1`, these conditions are automatically satisfied.

## Proposed Implementation

### 1. Class Structure

```python
# File: MarketData/ssvi_surface.py

class SSVIVolatilitySurface:
    """
    SSVI (Surface SVI) volatility surface across multiple maturities.
    
    Provides a parametric, arbitrage-free volatility surface that ensures
    consistency across different maturities.
    
    Attributes:
        forward: Forward price of the underlying
        maturities: Array of maturities in years
        atm_variances: ATM total variance at each maturity
        rho: Correlation parameter (constant across maturities)
        eta: Curvature parameter
        gamma: Power law exponent (optional, default=0.5)
    """
```

### 2. API Design

#### Initialization
```python
def __init__(
    self,
    forward: float,
    maturities: np.ndarray,
    atm_variances: np.ndarray,
    rho: float,
    eta: float,
    gamma: float = 0.5,
    curvature_function: str = "power_law"
):
    """
    Initialize SSVI volatility surface.
    
    Args:
        forward: Forward price
        maturities: Array of maturities (years), must be increasing
        atm_variances: ATM total variance at each maturity
        rho: Correlation parameter in (-1, 1)
        eta: Curvature parameter (> 0)
        gamma: Power law exponent in [0, 1] (default=0.5)
        curvature_function: Type of φ function ("power_law", "heston", "simple")
    """
```

#### Core Methods
```python
def get_volatility(
    self,
    strike: Union[float, np.ndarray],
    maturity: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Get implied volatility at given strike(s) and maturity(ies)."""

def get_variance(
    self,
    strike: Union[float, np.ndarray],
    maturity: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Get total implied variance at given strike(s) and maturity(ies)."""

def get_slice(self, maturity: float) -> SVIVolatilitySurface:
    """Extract a single-maturity SVI slice at given maturity."""

def validate_arbitrage_free(self) -> bool:
    """Check if the surface satisfies no-arbitrage conditions."""
```

#### Utility Methods
```python
def _curvature_function(self, theta: float) -> float:
    """Compute φ(θ) based on chosen parametrization."""

def _interpolate_atm_variance(self, maturity: float) -> float:
    """Interpolate ATM variance for arbitrary maturity."""

def _compute_raw_svi_params(self, maturity: float) -> Dict[str, float]:
    """Convert SSVI parameters to raw SVI parameters at given maturity."""
```

### 3. Integration with Existing Code

The SSVI implementation will integrate seamlessly with the existing architecture:

**Inheritance Approach (Option 1):**
```python
class SSVIVolatilitySurface(VolatilitySurface):
    """Extends base VolatilitySurface to support multiple maturities."""
```

**Composition Approach (Option 2):**
```python
class SSVIVolatilitySurface:
    """Standalone class that can generate SVIVolatilitySurface slices."""
    
    def get_slice(self, maturity: float) -> SVIVolatilitySurface:
        # Returns an SVIVolatilitySurface object for the given maturity
```

**Recommendation:** Use **Option 2** (composition) because:
- SSVI naturally operates across multiple maturities (different signature)
- Can generate single-maturity SVI slices when needed
- Cleaner separation of concerns
- More flexible for future extensions

### 4. File Organization

```
MarketData/
├── volatility_surface.py      # Base class (existing)
├── svi_surface.py             # Single-maturity SVI (existing)
├── ssvi_surface.py            # NEW: Multi-maturity SSVI
└── __init__.py                # Update imports
```

### 5. Parameter Calibration Support

While not part of core pricing, provide utilities to assist calibration:

```python
class SSVICalibrator:
    """Helper class for calibrating SSVI parameters to market data."""
    
    def calibrate(
        self,
        market_data: Dict[float, Dict[float, float]],
        initial_params: Dict[str, float],
        method: str = "least_squares"
    ) -> SSVIVolatilitySurface:
        """
        Calibrate SSVI surface to market implied volatilities.
        
        Args:
            market_data: {maturity: {strike: implied_vol}}
            initial_params: Initial guess for (rho, eta, gamma)
            method: Optimization method
        """
```

## Example Usage

### Basic Usage
```python
from MarketData.ssvi_surface import SSVIVolatilitySurface
import numpy as np

# Define maturities and ATM variances
maturities = np.array([0.25, 0.5, 1.0, 2.0])  # 3M, 6M, 1Y, 2Y
atm_variances = np.array([0.04, 0.08, 0.16, 0.32])  # θ_t = 0.16 * t

# Create SSVI surface
surface = SSVIVolatilitySurface(
    forward=100.0,
    maturities=maturities,
    atm_variances=atm_variances,
    rho=-0.4,           # Typical equity skew
    eta=1.5,            # Curvature parameter
    gamma=0.5           # Power law exponent
)

# Get volatility at specific strike and maturity
vol = surface.get_volatility(strike=95.0, maturity=1.0)

# Get volatilities across strikes for a given maturity
strikes = np.array([80, 90, 100, 110, 120])
vols = surface.get_volatility(strike=strikes, maturity=1.0)

# Extract SVI slice at specific maturity
svi_1y = surface.get_slice(maturity=1.0)
vol_from_slice = svi_1y.get_volatility(95.0)
```

### Advanced: Interpolation and Validation
```python
# Interpolate to maturity not in original grid
vol_9m = surface.get_volatility(strike=100.0, maturity=0.75)

# Validate arbitrage-free conditions
is_valid = surface.validate_arbitrage_free()
print(f"Surface is arbitrage-free: {is_valid}")

# Get full 2D grid of volatilities
strike_grid = np.linspace(70, 130, 50)
maturity_grid = np.linspace(0.25, 2.0, 20)
K, T = np.meshgrid(strike_grid, maturity_grid)
vol_surface = surface.get_volatility(K, T)  # 2D array
```

## Testing Strategy

### Unit Tests (`tests/test_marketdata.py`)

1. **Initialization Tests:**
   - Valid parameters
   - Invalid rho (outside [-1, 1])
   - Invalid eta (≤ 0)
   - Invalid gamma (outside [0, 1])
   - Non-increasing maturities
   - Non-increasing ATM variances (should warn/error)

2. **Mathematical Correctness:**
   - SSVI variance formula implementation
   - Curvature function computations (all types)
   - ATM variance interpolation
   - Conversion to raw SVI parameters
   - Consistency with SVI at single maturity

3. **Arbitrage-Free Conditions:**
   - Verify ∂w/∂t ≥ 0 numerically
   - Test calendar spread conditions
   - Validate butterfly arbitrage bounds

4. **Array Operations:**
   - Single strike, single maturity
   - Array strikes, single maturity
   - Single strike, array maturities
   - Array strikes, array maturities (broadcasting)

5. **Integration Tests:**
   - Extract SVI slice and verify consistency
   - Interpolate to intermediate maturities
   - Extrapolate beyond maturity range

### Example Test Structure
```python
class TestSSVIVolatilitySurface:
    def test_initialization(self):
        """Test basic initialization."""
        
    def test_power_law_curvature(self):
        """Test power law φ(θ) computation."""
        
    def test_variance_at_atm(self):
        """Verify ATM variance matches input."""
        
    def test_no_calendar_arbitrage(self):
        """Verify calendar spread conditions."""
        
    def test_consistency_with_svi_slice(self):
        """Check extracted slice matches direct SSVI evaluation."""
        
    def test_vectorization(self):
        """Test array input handling."""
```

## Implementation Phases

### Phase 1: Core Implementation (Week 1)
- [ ] Create `ssvi_surface.py` with basic class structure
- [ ] Implement SSVI variance formula
- [ ] Implement power law curvature function
- [ ] Add ATM variance interpolation (linear/cubic spline)
- [ ] Implement `get_variance()` and `get_volatility()` methods

### Phase 2: Validation & Testing (Week 1-2)
- [ ] Add parameter validation
- [ ] Implement no-arbitrage checks
- [ ] Write comprehensive unit tests (target: 20+ tests)
- [ ] Add integration tests with existing SVI

### Phase 3: Advanced Features (Week 2)
- [ ] Support for additional curvature functions (Heston, simple)
- [ ] Implement `get_slice()` method
- [ ] Add SVI parameter extraction
- [ ] Vectorization optimization

### Phase 4: Documentation & Examples (Week 2-3)
- [ ] Complete docstrings with mathematical references
- [ ] Create `example_ssvi_surface.py`
- [ ] Update README.md with SSVI section
- [ ] Add mathematical background to docs

### Phase 5: Calibration (Optional, Week 3+)
- [ ] Create `SSVICalibrator` helper class
- [ ] Implement least-squares calibration
- [ ] Add calibration examples and tests

## Dependencies

No new external dependencies required. Uses existing:
- `numpy` for numerical operations
- `scipy` (optional, for calibration via `scipy.optimize`)

## Performance Considerations

- Vectorize all operations using NumPy for efficiency
- Cache interpolated ATM variances for repeated queries at same maturity
- Pre-compute φ(θ) values for known maturities
- Consider using `numba` JIT compilation for critical paths (optional optimization)

## References

1. **Gatheral, J., & Jacquier, A. (2014).** "Arbitrage-free SVI volatility surfaces." 
   *Quantitative Finance*, 14(1), 59-71.
   - Original SSVI paper defining the parametrization

2. **Gatheral, J. (2004).** "A parsimonious arbitrage-free implied volatility parameterization 
   with application to the valuation of volatility derivatives."
   - Original SVI paper (already referenced in current implementation)

3. **Gatheral, J., & Jacquier, A. (2011).** "Convergence of Heston to SVI."
   *Quantitative Finance*, 11(8), 1129-1132.
   - Theoretical foundation for Heston-like curvature function

## Open Questions for Discussion

1. **Default Curvature Function:** Should we default to power law with γ=0.5, or make it mandatory?
2. **ATM Variance Input:** Accept as array, or provide helpers to construct from ATM vols?
3. **Extrapolation:** Should we allow extrapolation beyond maturity range, or raise error?
4. **Calibration:** Include in initial release or defer to separate PR?
5. **Backward Compatibility:** Any migration considerations from pure SVI users?

## Success Criteria

Implementation will be considered successful when:
- ✅ All unit tests pass (target: >95% code coverage)
- ✅ SSVI surface can be created and queried efficiently
- ✅ Extracted SVI slices are consistent with direct SSVI evaluation
- ✅ No-arbitrage validation works correctly
- ✅ Example code runs and produces expected output
- ✅ Documentation is complete and clear
- ✅ Performance is comparable to existing SVI (within 2x for single maturity)

## Conclusion

Adding SSVI to the quant-modelling library will significantly enhance its capabilities for modeling realistic volatility surfaces across multiple maturities. The proposed implementation:
- Maintains consistency with existing SVI architecture
- Follows established coding standards and patterns
- Provides comprehensive testing and validation
- Includes clear documentation and examples
- Enables arbitrage-free surface construction

The modular design allows for incremental development and testing, with clear milestones for each phase.

---

**Document Version:** 1.0  
**Author:** GitHub Copilot  
**Date:** January 16, 2026  
**Status:** Proposal - Awaiting Review
