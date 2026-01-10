"""Example usage of SSVI volatility surface.

This example demonstrates how to use the SSVI (Surface SVI) volatility surface
parametrization for modeling implied volatility across multiple maturities
in a calendar-arbitrage-free manner.
"""

import numpy as np
from MarketData.ssvi_surface import SSVIVolatilitySurface


def main():
    print("=" * 80)
    print("SSVI Volatility Surface Example")
    print("=" * 80)
    print()
    
    # Create an SSVI volatility surface with typical parameters
    # These parameters represent a typical equity volatility surface
    surface = SSVIVolatilitySurface(
        forward=100.0,           # Forward price of the underlying
        time_to_maturity=1.0,    # Reference maturity: 1 year
        a=0.04,                  # ATM variance level
        gamma=0.5,               # Power-law exponent for ATM variance
        rho=-0.6,                # Correlation (negative = downward skew)
        eta=1.0,                 # Curvature level parameter
        lambda_param=0.3         # Curvature power parameter
    )
    
    print("SSVI Surface Parameters:")
    print(f"  Forward: {surface.forward}")
    print(f"  Reference Maturity: {surface.time_to_maturity} years")
    print(f"  a (ATM variance level): {surface.a}")
    print(f"  γ (gamma, power parameter): {surface.gamma}")
    print(f"  ρ (correlation): {surface.rho}")
    print(f"  η (eta, curvature level): {surface.eta}")
    print(f"  λ (lambda, curvature power): {surface.lambda_param}")
    print()
    
    # Show how ATM variance evolves with maturity: θ(t) = a * t^γ
    print("ATM Total Variance Evolution (θ(t) = a * t^γ):")
    print(f"{'Maturity (years)':<20} {'θ(t)':<15} {'ATM Vol':<15}")
    print("-" * 50)
    
    maturities = [0.25, 0.5, 1.0, 2.0, 3.0]
    for T in maturities:
        theta = surface.get_theta(T)
        atm_vol = surface.get_volatility_at_maturity(100.0, T)
        print(f"{T:<20.2f} {theta:<15.6f} {atm_vol:<15.4f}")
    
    print()
    
    # Calculate implied volatilities across different strikes at 1-year maturity
    strikes = np.array([70, 80, 90, 95, 100, 105, 110, 120, 130])
    
    print("Implied Volatilities at 1-Year Maturity:")
    print(f"{'Strike':<10} {'Moneyness':<12} {'Volatility':<12} {'Variance':<12}")
    print("-" * 50)
    
    for strike in strikes:
        moneyness = strike / surface.forward
        vol = surface.get_volatility(strike)
        var = surface.get_variance(strike)
        print(f"{strike:<10.1f} {moneyness:<12.3f} {vol:<12.4f} {var:<12.6f}")
    
    print()
    
    # Demonstrate the volatility surface across multiple maturities and strikes
    print("Volatility Surface (Multiple Maturities):")
    print(f"{'Maturity\\Strike':<15}", end='')
    strike_display = [80, 90, 100, 110, 120]
    for s in strike_display:
        print(f"{s:<10}", end='')
    print()
    print("-" * 65)
    
    maturities_display = [0.25, 0.5, 1.0, 2.0, 3.0]
    for T in maturities_display:
        print(f"{T:<15.2f}", end='')
        for strike in strike_display:
            vol = surface.get_volatility_at_maturity(strike, T)
            print(f"{vol:<10.4f}", end='')
        print()
    
    print()
    
    # Show the volatility smile/skew effect at different maturities
    print("Volatility Smile/Skew Pattern Across Maturities:")
    print(f"{'Maturity':<10} {'K=70':<10} {'K=90':<10} {'K=100':<10} {'K=110':<10} {'K=130':<10}")
    print("-" * 60)
    
    for T in [0.25, 1.0, 3.0]:
        print(f"{T:<10.2f}", end='')
        for K in [70, 90, 100, 110, 130]:
            vol = surface.get_volatility_at_maturity(K, T)
            print(f"{vol:<10.4f}", end='')
        print()
    
    print()
    
    # Compare SSVI smile at different maturities to show term structure
    print("Comparison: Short-term vs Long-term Smile:")
    print(f"{'Strike':<10} {'Vol @ 3M':<12} {'Vol @ 1Y':<12} {'Vol @ 3Y':<12}")
    print("-" * 46)
    
    for strike in [70, 80, 90, 100, 110, 120, 130]:
        vol_3m = surface.get_volatility_at_maturity(strike, 0.25)
        vol_1y = surface.get_volatility_at_maturity(strike, 1.0)
        vol_3y = surface.get_volatility_at_maturity(strike, 3.0)
        print(f"{strike:<10.1f} {vol_3m:<12.4f} {vol_1y:<12.4f} {vol_3y:<12.4f}")
    
    print()
    
    # Demonstrate effect of correlation parameter across maturities
    print("Effect of Correlation Parameter (ρ) Across Maturities:")
    print(f"{'ρ':<8} {'Vol@K=80,T=1Y':<18} {'Vol@K=100,T=1Y':<18} {'Vol@K=120,T=1Y':<18}")
    print("-" * 62)
    
    for rho in [-0.8, -0.4, 0.0, 0.4, 0.8]:
        surface_rho = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            gamma=0.5,
            rho=rho,
            eta=1.0,
            lambda_param=0.3
        )
        vol_80 = surface_rho.get_volatility_at_maturity(80.0, 1.0)
        vol_100 = surface_rho.get_volatility_at_maturity(100.0, 1.0)
        vol_120 = surface_rho.get_volatility_at_maturity(120.0, 1.0)
        print(f"{rho:<8.1f} {vol_80:<18.4f} {vol_100:<18.4f} {vol_120:<18.4f}")
    
    print()
    
    # Show how curvature changes with gamma parameter
    print("Effect of Gamma Parameter (γ) on ATM Variance Term Structure:")
    print(f"{'γ':<8} {'θ(0.25Y)':<12} {'θ(1Y)':<12} {'θ(3Y)':<12}")
    print("-" * 44)
    
    for gamma in [0.3, 0.5, 0.7, 1.0]:
        surface_gamma = SSVIVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            a=0.04,
            gamma=gamma,
            rho=-0.6,
            eta=1.0,
            lambda_param=0.3
        )
        theta_025 = surface_gamma.get_theta(0.25)
        theta_1 = surface_gamma.get_theta(1.0)
        theta_3 = surface_gamma.get_theta(3.0)
        print(f"{gamma:<8.1f} {theta_025:<12.6f} {theta_1:<12.6f} {theta_3:<12.6f}")
    
    print()
    print("=" * 80)
    print("Key SSVI Features:")
    print("  • Calendar-arbitrage-free: θ(t) = a * t^γ ensures no arbitrage across time")
    print("  • Negative ρ creates downward skew (typical for equity markets)")
    print("  • Positive ρ creates upward skew")
    print("  • γ controls the term structure of ATM variance")
    print("  • λ controls how the smile curvature changes with maturity")
    print("=" * 80)


if __name__ == "__main__":
    main()
