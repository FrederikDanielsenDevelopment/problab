from dataclasses import dataclass

import numpy as np
from scipy.stats import beta

from src.problab.distributions._config import DEF_ALPHA
from src.problab.probability.intervals import ConfidenceInterval
from src.problab.statistics import clopper_pearson


@dataclass(frozen=True)
class ProbabilityResult:
    value: float
    num_successes: int
    num_unconditioned_samples: int
    num_conditioned_samples: int | None = None

    def __str__(self):
        return str(self.value)

    @property
    def num_samples(self) -> int:
        if self.num_conditioned_samples is None:
            return self.num_unconditioned_samples
        return self.num_conditioned_samples

    def confidence_interval( self,
                             alpha: float = DEF_ALPHA
                             ) -> ConfidenceInterval:

        return clopper_pearson.confidence_interval(
            num_samples=self.num_samples,
            num_successes=self.num_successes,
            alpha=alpha,
        )
