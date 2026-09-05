from numbers import Real

import numpy as np
from scipy.stats import norm

from src.problab.distributions.base import Distribution
from src.problab.random_variables.base import RandomVariable
from src.problab.random_variables.context import RealizationContext
from src.problab.random_variables.nodes import ConstantNode
from src.problab.value_sets import REALS, is_known_subset, POSITIVE_REALS, ValueSet

class NormalDistribution(Distribution):

    def __init__(self,
                 mean: RandomVariable | Real,
                 std: RandomVariable | Real
                 ) -> None:

        if isinstance(mean, RandomVariable):
            mean_node = mean._node
        elif isinstance(mean, Real):
            mean_node = ConstantNode(mean)
        else:
            raise TypeError("'mean' must be a RandomVariable or Real.")

        if not is_known_subset(mean_node.value_set, REALS):
            raise ValueError("'mean' must contain only real values.")

        self._mean = mean_node

        if isinstance(std, RandomVariable):
            std_node = std._node
        elif isinstance(std, Real):
            std_node = ConstantNode(std)
        else:
            raise TypeError("'std' must be a RandomVariable or Real.")

        if not is_known_subset(std_node.value_set, POSITIVE_REALS):
            raise ValueError("'std' must contain only positive real values.")

        self._std = std_node

        self._value_set = REALS

    @property
    def value_set(self) -> ValueSet:
        return self._value_set

    def sample(self, context: RealizationContext | None = None) -> np.ndarray:
        if context is None:
            context = RealizationContext()

        mean = context.evaluate(self._mean)
        std = context.evaluate(self._std)

        return norm.rvs(
            loc=mean,
            scale=std,
            size=context.num_samples,
            random_state=context.rng,
        )



