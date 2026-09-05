import numpy as np
from scipy.stats import beta

from src.problab.probability.intervals import ConfidenceInterval

def confidence_interval(
        num_samples: int,
        num_successes: int,
        alpha: float,
) -> ConfidenceInterval:

    if not 0 < alpha < 1:
        raise ValueError("'alpha' must be between 0 and 1.")

    if num_samples < 0:
        raise ValueError("'num_samples' must be non-negative.")

    if not 0 <= num_successes <= num_samples:
        raise ValueError(
            "'num_successes' must be between 0 and 'num_samples'."
        )

    n = num_samples
    k = num_successes

    if n == 0:
        return ConfidenceInterval(np.nan, np.nan, alpha)

    p_L = 0.0 if k == 0 else beta.ppf(
        q=alpha / 2,
        a=k,
        b=n - k + 1,
    )

    p_U = 1.0 if k == n else beta.ppf(
        q=1 - alpha / 2,
        a=k + 1,
        b=n - k,
    )

    return ConfidenceInterval(p_L, p_U, alpha)
