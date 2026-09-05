from numbers import Real
from typing import Iterable, Any

import numpy as np
import scipy as sp

from src.problab.distributions.base import Distribution
from src.problab.random_variables.context import RealizationContext
from src.problab.value_sets import ValueSet

class CategoricalDistribution(Distribution):

    def __init__(
            self,
            categories: Iterable[Any],
            probabilities: Iterable[float],
    ) -> None:
        self._categories = tuple(categories)
        self._probabilities = tuple(probabilities)

        if len(self._categories) == 0:
            raise ValueError("'categories' must contain at least one value.")

        if len(self._categories) != len(self._probabilities):
            raise ValueError("'categories' and 'probabilities' must have the same length.")

        if not all(isinstance(p, Real) for p in self._probabilities):
            raise TypeError("'probabilities' must contain only real numbers.")

        if not all(np.isfinite(p) for p in self._probabilities):
            raise ValueError("'probabilities' must be finite.")

        if any(p < 0 for p in self._probabilities):
            raise ValueError("'probabilities' cannot contain negative values.")

        if not np.isclose(sum(self._probabilities), 1.0):
            raise ValueError("'probabilities' must sum to 1.")

        self._value_set = sp.FiniteSet(*self._categories)

    @property
    def value_set(self) -> ValueSet:
        return self._value_set

    def sample(self, context: RealizationContext | None = None) -> np.ndarray:

        if context is None:
            num_samples = 1
            rng = np.random.default_rng()
        else:
            num_samples = context.num_samples
            rng = context.rng

        return np.array([
            self._categories[i]
            for i in rng.choice(
                len(self._categories),
                size=num_samples,
                p=self._probabilities
            )])


