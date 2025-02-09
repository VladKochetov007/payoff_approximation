from math import log, sqrt, exp
from scipy.stats import norm

def black_scholes_price(
    S: float,         # Current asset price
    K: float,         # Strike price
    T: float,         # Time to maturity (in years)
    r: float,         # Risk-free interest rate
    sigma: float,     # Asset volatility
    option_type: str  # Option type: 'call' or 'put'
) -> float:
    """
    Calculates the price of a European option using the Black-Scholes model.

    Parameters:
    S: Current asset price
    K: Strike price
    T: Time to maturity
    r: Risk-free interest rate
    sigma: Asset volatility
    option_type: Option type ('call' or 'put')

    Returns:
    Option price
    """
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Unsupported option type. Use 'call' or 'put'.")
    
    return price


def black_scholes_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> dict[str, float]:
    """
    Calculates the Greeks for a European option using the Black-Scholes model.

    Parameters:
    S: Current asset price
    K: Strike price
    T: Time to maturity
    r: Risk-free interest rate
    sigma: Asset volatility
    option_type: Option type ('call' or 'put')

    Returns:
    Dictionary with Greeks: delta, gamma, vega, theta, rho
    """
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    
    delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
    vega = S * norm.pdf(d1) * sqrt(T)
    theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt(T)) 
             - r * K * exp(-r * T) * norm.cdf(d2)) if option_type == 'call' else (
             -S * norm.pdf(d1) * sigma / (2 * sqrt(T)) 
             + r * K * exp(-r * T) * norm.cdf(-d2))
    rho = K * T * exp(-r * T) * norm.cdf(d2) if option_type == 'call' else (
          -K * T * exp(-r * T) * norm.cdf(-d2))
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }
