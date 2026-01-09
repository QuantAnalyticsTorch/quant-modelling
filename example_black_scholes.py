"""Example usage of the Black-Scholes pricer.

This script demonstrates how to use the Black model with the analytical calculator
to price European options and calculate their Greeks.
"""

from Models.black import BlackModel
from Securities.european_option import EuropeanOption
from Calculators.analytical import AnalyticalCalculator


def main():
    """Demonstrate Black-Scholes option pricing."""
    print("=" * 60)
    print("Black-Scholes Option Pricer Example")
    print("=" * 60)
    
    # Create a Black model
    forward = 100.0
    volatility = 0.2  # 20% annual volatility
    discount_factor = 0.95  # 5% discount rate
    
    model = BlackModel(
        forward=forward,
        volatility=volatility,
        discount_factor=discount_factor
    )
    
    print(f"\nModel Parameters:")
    print(f"  Forward Price: ${forward:.2f}")
    print(f"  Volatility: {volatility * 100:.1f}%")
    print(f"  Discount Factor: {discount_factor:.3f}")
    
    # Create European call and put options
    strike = 100.0
    maturity = 1.0  # 1 year
    
    call = EuropeanOption(
        strike=strike,
        maturity=maturity,
        option_type='call'
    )
    
    put = EuropeanOption(
        strike=strike,
        maturity=maturity,
        option_type='put'
    )
    
    print(f"\nOption Parameters:")
    print(f"  Strike Price: ${strike:.2f}")
    print(f"  Maturity: {maturity:.1f} year(s)")
    
    # Create analytical calculator
    calculator = AnalyticalCalculator()
    
    # Price the options
    print("\n" + "=" * 60)
    print("Call Option Pricing")
    print("=" * 60)
    
    call_price = calculator.price(model, call)
    print(f"\nCall Price: ${call_price:.4f}")
    
    call_greeks = calculator.greeks(model, call)
    print(f"\nCall Greeks:")
    print(f"  Delta: {call_greeks['delta']:.4f}")
    print(f"  Gamma: {call_greeks['gamma']:.4f}")
    print(f"  Vega: {call_greeks['vega']:.4f}")
    print(f"  Theta: {call_greeks['theta']:.4f}")
    
    # Price the put option
    print("\n" + "=" * 60)
    print("Put Option Pricing")
    print("=" * 60)
    
    put_price = calculator.price(model, put)
    print(f"\nPut Price: ${put_price:.4f}")
    
    put_greeks = calculator.greeks(model, put)
    print(f"\nPut Greeks:")
    print(f"  Delta: {put_greeks['delta']:.4f}")
    print(f"  Gamma: {put_greeks['gamma']:.4f}")
    print(f"  Vega: {put_greeks['vega']:.4f}")
    print(f"  Theta: {put_greeks['theta']:.4f}")
    
    # Verify put-call parity
    print("\n" + "=" * 60)
    print("Put-Call Parity Verification")
    print("=" * 60)
    
    parity_lhs = call_price - put_price
    parity_rhs = discount_factor * (forward - strike)
    
    print(f"\nC - P = ${parity_lhs:.6f}")
    print(f"D × (F - K) = ${parity_rhs:.6f}")
    print(f"Difference: ${abs(parity_lhs - parity_rhs):.10f}")
    
    if abs(parity_lhs - parity_rhs) < 1e-10:
        print("\n✓ Put-call parity holds!")
    else:
        print("\n✗ Put-call parity violated!")
    
    # Example with different strikes
    print("\n" + "=" * 60)
    print("Option Prices at Different Strikes")
    print("=" * 60)
    
    strikes = [80, 90, 100, 110, 120]
    print(f"\n{'Strike':<10} {'Call Price':<15} {'Put Price':<15}")
    print("-" * 40)
    
    for K in strikes:
        call_K = EuropeanOption(strike=K, maturity=maturity, option_type='call')
        put_K = EuropeanOption(strike=K, maturity=maturity, option_type='put')
        
        call_price_K = calculator.price(model, call_K)
        put_price_K = calculator.price(model, put_K)
        
        print(f"${K:<9.2f} ${call_price_K:<14.4f} ${put_price_K:<14.4f}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
