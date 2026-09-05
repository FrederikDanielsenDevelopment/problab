from dataclasses import dataclass
import numpy as np
from scipy.stats import binom


@dataclass(frozen=True)
class ConfidenceInterval:
    lower: float
    upper: float
    alpha: float

@dataclass(frozen=True)
class ProbabilityInterval():
    lower: float
    upper: float
    alpha: float
    is_estimate: bool

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError("'alpha' must be between 0 and 1.")

        if self.lower > self.upper:
            raise ValueError("'lower' must be less than 'upper'.")

    @property
    def probability(self) -> float:
        return 1 - self.alpha

    def __str__(self) -> str:
        return f"[{self.lower}, {self.upper}]"
