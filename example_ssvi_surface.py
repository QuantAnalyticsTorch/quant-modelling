"""Example usage of SSVI volatility surface.

This example demonstrates how to use the SSVI (Surface SVI)
volatility surface parametrization for modeling implied volatility surfaces
across maturities in an arbitrage-free manner.
"""

import numpy as np
from MarketData.ssvi_surface import SSVIVolatilitySurface


def main():
    print("=" * 70)
    print("SSVI Volatility Surface Example")
    print("=" * 70)
    print()
    
    # Create an SSVI volatility surface with typical parameters
    # These parameters represent a typical equity volatility surface
    surface = SSVIVolatilitySurface(
        forward=100.0,           # Forward price of the underlying
        time_to_maturity=1.0,    # 1 year to maturity
        theta=0.04,              # ATM total variance
        rho=-0.3,                # Correlation (negative = downward skew)
        gamma=0.5,               # Power-law exponent
        eta=1.5                  # Scale parameter
    )
    
    print("SSVI Surface Parameters:")
    print(f"  Forward: {surface.forward}")
    print(f"  Time to Maturity: {surface.time_to_maturity} years")
    print(f"  θ (ATM variance): {surface.theta}")
    print(f"  ρ (correlation): {surface.rho}")
    print(f"  γ (gamma): {surface.gamma}")
    print(f"  η (eta): {surface.eta}")
    print(f"  φ(θ) = η/θ^γ: {surface._phi():.4f}")
    print()
    
    # Calculate implied volatilities across different strikes
    strikes = np.array([70, 80, 90, 95, 100, 105, 110, 120, 130])
    
    print("Implied Volatilities across Strikes:")
    print(f"{'Strike':<10} {'Moneyness':<12} {'Volatility':<12} {'Variance':<12}")
    print("-" * 50)
    
    for strike in strikes:
        moneyness = strike / surface.forward
        vol = surface.get_volatility(strike)
        var = surface.get_variance(strike)
        print(f"{strike:<10.1f} {moneyness:<12.3f} {vol:<12.4f} {var:<12.6f}")
    
    print()
    
    # Demonstrate vectorized computation
    print("Vectorized Computation:")
    strikes_vector = np.linspace(70, 130, 13)
    vols_vector = surface.get_volatility(strikes_vector)
    
    print(f"Number of strikes: {len(strikes_vector)}")
    print(f"Min volatility: {vols_vector.min():.4f}")
    print(f"Max volatility: {vols_vector.max():.4f}")
    print(f"ATM volatility (K=100): {surface.get_volatility(100.0):.4f}")
    print()
    
    # Show the volatility smile/skew effect
    print("Volatility Smile/Skew Pattern:")
    print(f"  Deep ITM Put  (K=70):  vol = {surface.get_volatility(70.0):.4f}")
    print(f"  ITM Put       (K=90):  vol = {surface.get_volatility(90.0):.4f}")
    print(f"  ATM          (K=100):  vol = {surface.get_volatility(100.0):.4f}")
    print(f"  OTM Call     (K=110):  vol = {surface.get_volatility(110.0):.4f}")
    print(f"  Deep OTM Call (K=130): vol = {surface.get_volatility(130.0):.4f}")
    print()
    
    # Demonstrate how the volatility smile changes with different gamma values
    print("Effect of Gamma Parameter (γ):")
    print(f"{'γ':<10} {'Vol@K=80':<12} {'Vol@K=100':<12} {'Vol@K=120':<12}")
    print("-" * 46)
    
    for gamma in [0.0, 0.2, 0.4, 0.6, 0.8]:
        # Adjust rho to satisfy arbitrage-free constraint: |rho| <= (1-gamma)/(1+gamma)
        if gamma > 0:
            max_rho = (1 - gamma) / (1 + gamma)
            rho = -min(0.4, max_rho * 0.9)  # Use 90% of max to stay safe
        else:
            rho = -0.4
        
        surface_gamma = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            theta=0.04,
            rho=rho,
            gamma=gamma,
            eta=1.5
        )
        vol_80 = surface_gamma.get_volatility(80.0)
        vol_100 = surface_gamma.get_volatility(100.0)
        vol_120 = surface_gamma.get_volatility(120.0)
        print(f"{gamma:<10.1f} {vol_80:<12.4f} {vol_100:<12.4f} {vol_120:<12.4f}")
    
    print()
    
    # Show SSVI surfaces for different maturities
    print("SSVI Surfaces for Different Maturities:")
    print(f"{'Maturity':<12} {'θ':<12} {'ATM Vol':<12} {'Vol@K=80':<12} {'Vol@K=120':<12}")
    print("-" * 60)
    
    maturities = [0.25, 0.5, 1.0, 2.0]
    # Typical term structure: theta increases with maturity
    thetas = [0.01, 0.02, 0.04, 0.08]
    
    for T, theta in zip(maturities, thetas):
        surface_T = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=T,
            theta=theta,
            rho=-0.3,
            gamma=0.5,
            eta=1.5
        )
        vol_atm = surface_T.get_volatility(100.0)
        vol_80 = surface_T.get_volatility(80.0)
        vol_120 = surface_T.get_volatility(120.0)
        print(f"{T:<12.2f} {theta:<12.4f} {vol_atm:<12.4f} {vol_80:<12.4f} {vol_120:<12.4f}")
    
    print()
    print("=" * 70)
    print("Key Features of SSVI:")
    print("  • Arbitrage-free parametrization across maturities")
    print("  • θ(T) controls the ATM variance term structure")
    print("  • Negative ρ creates downward skew (typical for equity markets)")
    print("  • γ and η control the wing behavior")
    print("  • Constraint: |ρ| ≤ (1-γ)/(1+γ) for arbitrage-free surfaces")
    print("=" * 70)


if __name__ == "__main__":
    main()
