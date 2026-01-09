"""Example usage of SABR volatility surface.

This example demonstrates how to use the SABR (Stochastic Alpha, Beta, Rho)
volatility surface parametrization for modeling implied volatility smiles
in different markets (FX and interest rates).
"""

import numpy as np
from MarketData.sabr_surface import SABRVolatilitySurface


def main():
    print("=" * 70)
    print("SABR Volatility Surface Example")
    print("=" * 70)
    print()
    
    # Example 1: Lognormal SABR (β = 1) - typical for FX markets
    print("Example 1: Lognormal SABR (β = 1) - FX Market")
    print("-" * 70)
    
    surface_fx = SABRVolatilitySurface(
        forward=100.0,           # Forward price of the currency pair
        time_to_maturity=1.0,    # 1 year to maturity
        alpha=0.25,              # Initial volatility (25%)
        beta=1.0,                # Lognormal model (typical for FX)
        rho=-0.3,                # Negative correlation (downward skew)
        nu=0.4                   # Volatility of volatility (40%)
    )
    
    print("SABR Parameters:")
    print(f"  Forward: {surface_fx.forward}")
    print(f"  Time to Maturity: {surface_fx.time_to_maturity} years")
    print(f"  α (alpha): {surface_fx.alpha}")
    print(f"  β (beta): {surface_fx.beta} (Lognormal)")
    print(f"  ρ (rho): {surface_fx.rho}")
    print(f"  ν (nu): {surface_fx.nu}")
    print()
    
    # Calculate implied volatilities across different strikes
    strikes_fx = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
    
    print("Implied Volatilities across Strikes:")
    print(f"{'Strike':<10} {'Moneyness':<12} {'Volatility':<12} {'Variance':<12}")
    print("-" * 50)
    
    for strike in strikes_fx:
        moneyness = strike / surface_fx.forward
        vol = surface_fx.get_volatility(strike)
        var = surface_fx.get_variance(strike)
        print(f"{strike:<10.1f} {moneyness:<12.3f} {vol:<12.4f} {var:<12.6f}")
    
    print()
    
    # Show the volatility smile pattern
    print("FX Volatility Smile Pattern:")
    print(f"  Deep ITM Put  (K=80):  vol = {surface_fx.get_volatility(80.0):.4f}")
    print(f"  ITM Put       (K=90):  vol = {surface_fx.get_volatility(90.0):.4f}")
    print(f"  ATM          (K=100):  vol = {surface_fx.get_volatility(100.0):.4f}")
    print(f"  OTM Call     (K=110):  vol = {surface_fx.get_volatility(110.0):.4f}")
    print(f"  Deep OTM Call (K=120): vol = {surface_fx.get_volatility(120.0):.4f}")
    print()
    print()
    
    # Example 2: Normal SABR (β = 0) - typical for interest rate markets
    print("Example 2: Normal SABR (β = 0) - Interest Rate Market")
    print("-" * 70)
    
    surface_ir = SABRVolatilitySurface(
        forward=0.03,            # 3% interest rate
        time_to_maturity=5.0,    # 5 years to maturity
        alpha=0.0050,            # 50 bps normal volatility
        beta=0.0,                # Normal model (typical for rates)
        rho=0.0,                 # No correlation
        nu=0.2                   # Volatility of volatility (20%)
    )
    
    print("SABR Parameters:")
    print(f"  Forward Rate: {surface_ir.forward * 100:.2f}%")
    print(f"  Time to Maturity: {surface_ir.time_to_maturity} years")
    print(f"  α (alpha): {surface_ir.alpha} ({surface_ir.alpha * 10000:.0f} bps)")
    print(f"  β (beta): {surface_ir.beta} (Normal)")
    print(f"  ρ (rho): {surface_ir.rho}")
    print(f"  ν (nu): {surface_ir.nu}")
    print()
    
    # Calculate volatilities for different strike rates
    strikes_ir = np.array([0.02, 0.025, 0.03, 0.035, 0.04])
    
    print("Interest Rate Volatilities across Strikes:")
    print(f"{'Strike (%)':<12} {'Vol (bps)':<15} {'Variance':<15}")
    print("-" * 45)
    
    for strike in strikes_ir:
        vol = surface_ir.get_volatility(strike)
        var = surface_ir.get_variance(strike)
        # Convert to basis points for display
        vol_bps = vol * 10000
        print(f"{strike * 100:<12.2f} {vol_bps:<15.2f} {var:<15.8f}")
    
    print()
    print()
    
    # Example 3: CIR SABR (β = 0.5) - common for commodities
    print("Example 3: CIR SABR (β = 0.5) - Commodity Market")
    print("-" * 70)
    
    surface_commodity = SABRVolatilitySurface(
        forward=50.0,            # Forward price (e.g., oil)
        time_to_maturity=0.5,    # 6 months to maturity
        alpha=0.30,              # 30% volatility
        beta=0.5,                # CIR/square-root model
        rho=-0.4,                # Negative correlation
        nu=0.5                   # Volatility of volatility
    )
    
    print("SABR Parameters:")
    print(f"  Forward: {surface_commodity.forward}")
    print(f"  Time to Maturity: {surface_commodity.time_to_maturity} years")
    print(f"  α (alpha): {surface_commodity.alpha}")
    print(f"  β (beta): {surface_commodity.beta} (CIR)")
    print(f"  ρ (rho): {surface_commodity.rho}")
    print(f"  ν (nu): {surface_commodity.nu}")
    print()
    
    # Calculate volatilities
    strikes_commodity = np.array([40, 45, 50, 55, 60])
    vols_commodity = surface_commodity.get_volatility(strikes_commodity)
    
    print("Commodity Volatilities:")
    print(f"{'Strike':<10} {'Volatility':<12}")
    print("-" * 25)
    for strike, vol in zip(strikes_commodity, vols_commodity):
        print(f"{strike:<10.1f} {vol:<12.4f}")
    
    print()
    print()
    
    # Demonstrate the effect of different beta values
    print("Effect of Beta Parameter on ATM Volatility:")
    print(f"{'β':<10} {'Vol@ATM':<15} {'Model Type':<20}")
    print("-" * 50)
    
    for beta_val, model_type in [(0.0, "Normal"), (0.25, "Mixed"), 
                                  (0.5, "CIR"), (0.75, "Mixed"), 
                                  (1.0, "Lognormal")]:
        surface_beta = SABRVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            alpha=0.20,
            beta=beta_val,
            rho=-0.2,
            nu=0.3
        )
        vol_atm = surface_beta.get_volatility(100.0)
        print(f"{beta_val:<10.2f} {vol_atm:<15.4f} {model_type:<20}")
    
    print()
    print()
    
    # Demonstrate the effect of rho on the smile
    print("Effect of Correlation (ρ) on Volatility Smile:")
    print(f"{'ρ':<10} {'Vol@K=80':<12} {'Vol@K=100':<12} {'Vol@K=120':<12}")
    print("-" * 50)
    
    for rho_val in [-0.8, -0.4, 0.0, 0.4, 0.8]:
        surface_rho = SABRVolatilitySurface(
            forward=100.0,
            time_to_maturity=1.0,
            alpha=0.25,
            beta=0.7,
            rho=rho_val,
            nu=0.4
        )
        vol_80 = surface_rho.get_volatility(80.0)
        vol_100 = surface_rho.get_volatility(100.0)
        vol_120 = surface_rho.get_volatility(120.0)
        print(f"{rho_val:<10.1f} {vol_80:<12.4f} {vol_100:<12.4f} {vol_120:<12.4f}")
    
    print()
    print()
    
    # Demonstrate vectorized computation
    print("Vectorized Computation Example:")
    strikes_vector = np.linspace(70, 130, 25)
    vols_vector = surface_fx.get_volatility(strikes_vector)
    
    print(f"Number of strikes computed: {len(strikes_vector)}")
    print(f"Min volatility: {vols_vector.min():.4f}")
    print(f"Max volatility: {vols_vector.max():.4f}")
    print(f"ATM volatility (K=100): {surface_fx.get_volatility(100.0):.4f}")
    
    print()
    print("=" * 70)
    print("Notes:")
    print("  • β = 0: Normal SABR (typical for interest rates)")
    print("  • β = 0.5: CIR/Square-root SABR (typical for commodities)")
    print("  • β = 1: Lognormal SABR (typical for FX and equities)")
    print("  • Negative ρ creates downward skew")
    print("  • Positive ρ creates upward skew")
    print("  • ν controls the curvature of the smile")
    print("=" * 70)


if __name__ == "__main__":
    main()
