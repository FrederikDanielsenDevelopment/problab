from numbers import Real

import numpy as np
import scipy as sp
from scipy.stats import binom

from src.problab.distributions.base import Distribution
from src.problab.random_variables.base import RandomVariable
from src.problab.random_variables.context import RealizationContext
from src.problab.random_variables.nodes import ConstantNode
from src.problab.value_sets import ValueSet, NATURALS_0, is_known_subset, UNIT_INTERVAL

class BinomialDistribution(Distribution):

    def __init__(
            self,
            n: RandomVariable | int,
            p: RandomVariable | Real
        ) -> None:

        if isinstance(n, RandomVariable):
            n_node = n._node
        elif isinstance(n, int):
            n_node = ConstantNode(n)
        else:
            raise TypeError("'n' must be a RandomVariable or int.")

        if not is_known_subset(n_node.value_set, NATURALS_0):
            raise ValueError("'n' must be a positive integer or 0.")

        self._n = n_node

        n_max = n_node.value_set.sup

        if n_max == sp.oo:
            self._value_set = NATURALS_0
        else:
            self._value_set = sp.FiniteSet(*range(int(n_max) + 1))

        if isinstance(p, RandomVariable):
            p_node = p._node
        elif isinstance(p, Real):
            p_node = ConstantNode(p)
        else:
            raise TypeError("'p' must be a RandomVariable or Real.")

        if not is_known_subset(p_node.value_set, UNIT_INTERVAL):
            raise ValueError("'p' must be in the interval [0, 1].")

        self._p = p_node


    @property
    def value_set(self) -> ValueSet:
        return self._value_set

    def sample(self, context: RealizationContext | None = None) -> np.ndarray:
        if context is None:
            context = RealizationContext()

        n = context.evaluate(self._n)
        p = context.evaluate(self._p)

        return binom.rvs(
            n=n,
            p=p,
            size=context.num_samples,
            random_state=context.rng
        )
