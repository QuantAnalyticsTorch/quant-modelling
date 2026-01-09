"""Statistical distribution functions for quantitative modeling."""

import numpy as np
from scipy.special import erf
from typing import Union


def normal_cdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the cumulative distribution function of the standard normal distribution.
    
    Uses the error function approximation for numerical stability.
    
    Args:
        x: Input value(s) at which to evaluate the CDF.
        
    Returns:
        The cumulative probability N(x) for standard normal distribution.
        
    Examples:
        >>> normal_cdf(0.0)
        0.5
        >>> normal_cdf(1.96)  # doctest: +ELLIPSIS
        0.975...
    """
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def normal_pdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the probability density function of the standard normal distribution.
    
    Args:
        x: Input value(s) at which to evaluate the PDF.
        
    Returns:
        The probability density φ(x) for standard normal distribution.
        
    Examples:
        >>> abs(normal_pdf(0.0) - 0.3989423) < 1e-6
        True
        >>> normal_pdf(0.0) > normal_pdf(1.0)
        True
    """
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
