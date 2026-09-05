import numpy as np
from scipy.stats import binom
from typing import Literal, TypeAlias

from src.problab.distributions._config import DEF_ALPHA
from src.problab.probability.intervals import ConfidenceInterval

QuantileMethod: TypeAlias = Literal[
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
]


def quantile_confidence_interval(samples: np.ndarray,
                                 q: float,
                                 alpha: float = DEF_ALPHA
                                 ) -> ConfidenceInterval:

    if not 0 < q < 1:
        raise ValueError("'q' must be between 0 and 1.")

    if not 0 < alpha < 1:
        raise ValueError("'alpha' must be between 0 and 1.")

    n = len(samples)

    if n == 0:
        raise ValueError("'samples' must contain at least one value.")

    k_L = 1

    while (
            k_L < n - 1
            and binom.cdf(k_L, n, q) <= alpha / 2
    ):
        k_L += 1

    k_U = n - 1

    while (
            k_U > 1
            and 1 - binom.cdf(k_U - 1, n, q) <= alpha / 2
    ):
        k_U -= 1

    sorted_samples = np.sort(samples)

    lower = float(sorted_samples[k_L - 1])
    upper = float(sorted_samples[k_U])

    return ConfidenceInterval(
        lower=lower,
        upper=upper,
        alpha=alpha,
    )