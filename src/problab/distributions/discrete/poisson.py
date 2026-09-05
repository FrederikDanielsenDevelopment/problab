from numbers import Real

import numpy as np
from scipy.stats import poisson

from src.problab.distributions.base import Distribution
from src.problab.random_variables.base import RandomVariable
from src.problab.random_variables.context import RealizationContext
from src.problab.random_variables.nodes import ConstantNode
from src.problab.value_sets import ValueSet, NATURALS_0, is_known_subset, NON_NEGATIVE_REALS

class PoissonDistribution(Distribution):

    def __init__(
            self,
            mu: RandomVariable | Real,
        ) -> None:

        if isinstance(mu, RandomVariable):
            mu_node = mu._node
        elif isinstance(mu, Real):
            mu_node = ConstantNode(mu)
        else:
            raise TypeError("'mu' must be a RandomVariable or real.")

        if not is_known_subset(mu_node.value_set, NON_NEGATIVE_REALS):
            raise ValueError("'mu' must be non-negative.")

        self._mu = mu_node
        self._value_set = NATURALS_0

    @property
    def value_set(self) -> ValueSet:
        return self._value_set

    def sample(self, context: RealizationContext | None = None) -> np.ndarray:
        if context is None:
            context = RealizationContext()

        mu = context.evaluate(self._mu)

        return poisson.rvs(
            mu=mu,
            size=context.num_samples,
            random_state=context.rng
        )
