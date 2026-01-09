"""Example usage of SVI volatility surface.

This example demonstrates how to use the SVI (Stochastic Volatility Inspired)
volatility surface parametrization for modeling implied volatility smiles.
"""

import numpy as np
from MarketData.svi_surface import SVIVolatilitySurface


def main():
    print("=" * 70)
    print("SVI Volatility Surface Example")
    print("=" * 70)
    print()
    
    # Create an SVI volatility surface with typical parameters
    # These parameters represent a typical equity volatility smile
    surface = SVIVolatilitySurface(
        forward=100.0,           # Forward price of the underlying
        time_to_maturity=1.0,    # 1 year to maturity
        a=0.04,                  # Vertical shift (ATM variance level)
        b=0.15,                  # Slope (variance sensitivity)
        rho=-0.6,                # Correlation (negative = downward skew)
        m=0.0,                   # Horizontal shift (ATM location)
        sigma=0.3                # Curvature (smile shape)
    )
    
    print("SVI Surface Parameters:")
    print(f"  Forward: {surface.forward}")
    print(f"  Time to Maturity: {surface.time_to_maturity} years")
    print(f"  a (vertical shift): {surface.a}")
    print(f"  b (slope): {surface.b}")
    print(f"  ρ (correlation): {surface.rho}")
    print(f"  m (horizontal shift): {surface.m}")
    print(f"  σ (curvature): {surface.sigma}")
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
    
    # Demonstrate how the volatility smile changes with different rho values
    print("Effect of Correlation Parameter (ρ):")
    print(f"{'ρ':<10} {'Vol@K=80':<12} {'Vol@K=100':<12} {'Vol@K=120':<12}")
    print("-" * 46)
    
    for rho in [-0.8, -0.4, 0.0, 0.4, 0.8]:
        surface_rho = SVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            b=0.15,
            rho=rho,
            m=0.0,
            sigma=0.3
        )
        vol_80 = surface_rho.get_volatility(80.0)
        vol_100 = surface_rho.get_volatility(100.0)
        vol_120 = surface_rho.get_volatility(120.0)
        print(f"{rho:<10.1f} {vol_80:<12.4f} {vol_100:<12.4f} {vol_120:<12.4f}")
    
    print()
    print("=" * 70)
    print("Note: Negative ρ creates downward skew (typical for equity markets)")
    print("      Positive ρ creates upward skew")
    print("=" * 70)


if __name__ == "__main__":
    main()
