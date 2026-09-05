from __future__ import annotations

from itertools import count
from numbers import Real, Complex
from typing import Callable, TypeVar
import numpy as np
import sympy as sp

from src.problab.distributions._config import DEF_NUM_SAMPLES, DEF_ALPHA
from src.problab.distributions.base import Distribution
from src.problab.events import Event
from src.problab.operations import ADD, SUBTRACT, MULTIPLY, MODULO, LT, LTE, GT, GTE, EQ, NEQ
from src.problab.probability.intervals import ProbabilityInterval, ConfidenceInterval
from src.problab.random_variables.context import RealizationContext
from src.problab.random_variables.nodes import DistributionNode, Node, ConstantNode, OperationNode
from src.problab.statistics.quantiles import quantile_confidence_interval
from src.problab.value_sets import is_known_subset, COMPLEXES, REALS, ValueSet

class RandomVariable:

    _count = count()

    def __init__(
            self,
            distribution: Distribution,
            name: str | None = None,
    ) -> None:
        self._distribution = distribution

        if name is None: self._name = "RV_" + str(next(RandomVariable._count) + 1)
        else: self._name = name

        self._node = DistributionNode(distribution)

    @classmethod
    def _from_node(
            cls,
            node: Node,
            name: str | None = None,
    ) -> RandomVariable:
        rv = cls.__new__(cls)

        rv._node = node
        rv._name = name if name is not None else "RV_" + str(next(cls._count) + 1)

        return rv

    @property
    def name(self) -> str:
        return self._name

    def realize(self):
        return self.sample()

    def sample(self, num_samples: int = 1,  rng: np.random.Generator | None = None,) -> np.ndarray:

        if num_samples < 1:
            raise ValueError("Number of samples must be positive")

        return RealizationContext(num_samples, rng).evaluate(self._node)

    def _is_real_or_complex(self) -> bool:
        return is_known_subset(self._node.value_set, COMPLEXES)

    def _binary_operation(
            self,
            other: RandomVariable | Complex,
            operation: Callable,
            reverse: bool = False,
    ) -> RandomVariable:

        if not self._is_real_or_complex():
            return NotImplemented

        if not isinstance(other, (RandomVariable, Complex)):
            return NotImplemented

        if isinstance(other, RandomVariable):
            if not other._is_real_or_complex():
                return NotImplemented
        else:
            other = RandomVariable._from_node(
                ConstantNode(other)
            )

        result_value_set = (
            REALS
            if is_known_subset(self._node.value_set, REALS)
               and is_known_subset(other._node.value_set, REALS)
            else COMPLEXES
        )

        inputs = (
            (other._node, self._node)
            if reverse
            else (self._node, other._node)
        )

        return RandomVariable._from_node(
            OperationNode(
                operation=operation,
                inputs=inputs,
                value_set=result_value_set,
            )
        )

    def interval(self,
                 alpha: float,
                 num_samples: int | None = None,
                 rng: np.random.Generator | None = None
                 ) -> ProbabilityInterval:

        if not is_known_subset(self._node.value_set, REALS):
            raise TypeError("Probability intervals are only defined for real-valued random variables." )

        if not 0 < alpha < 1:
            raise ValueError("'alpha' must be between 0 and 1.")

        samples = self.sample(
            num_samples=num_samples,
            rng=rng,
        )

        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2

        lower = float(np.quantile(samples, lower_quantile))
        upper = float(np.quantile(samples, upper_quantile))

        return ProbabilityInterval(
            lower=lower,
            upper=upper,
            alpha=alpha,
            is_estimate=True
        )

    def apply(
            self,
            function: Callable,
            *others: RandomVariable,
            value_set: ValueSet,
            vectorized: bool = False,
    ) -> RandomVariable:

        if not callable(function):
            raise TypeError("'function' must be callable.")

        if not all(isinstance(other, RandomVariable) for other in others):
            raise TypeError("'others' must be RandomVariable instances.")

        inputs = (
            self._node,
            *(other._node for other in others),
        )

        if vectorized:
            operation = function
        else:
            operation = lambda *values: np.asarray([
                function(*realization)
                for realization in zip(*values)
            ])

        return RandomVariable._from_node(
            OperationNode(
                operation=operation,
                inputs=inputs,
                value_set=value_set,
            )
        )


    def __add__(self, other):
        return self._binary_operation(other, ADD)

    def __radd__(self, other):
        return self._binary_operation(other, ADD)

    def __sub__(self, other):
        return self._binary_operation(other, SUBTRACT)

    def __rsub__(self, other):
        return self._binary_operation(other, SUBTRACT, reverse=True)

    def __mul__(self, other):
        return self._binary_operation(other, MULTIPLY)

    def __rmul__(self, other):
        return self._binary_operation(other, MULTIPLY)

    def __truediv__(self, other, DIVIDE=None):
        return self._binary_operation(other, DIVIDE)

    def __rtruediv__(self, other, DIVIDE=None):
        return self._binary_operation(other, DIVIDE, reverse=True)

    def __mod__(self, other: RandomVariable | Real) -> RandomVariable:

        if not is_known_subset(self._node.value_set, REALS):
            return NotImplemented

        if not isinstance(other, (RandomVariable, Real)):
            return NotImplemented

        if isinstance(other, RandomVariable):
            if not is_known_subset(other._node.value_set, REALS):
                return NotImplemented
        else:
            other = RandomVariable._from_node(
                ConstantNode(other)
            )

        return RandomVariable._from_node(
            OperationNode(
                operation=MODULO,
                inputs=(self._node, other._node),
                value_set=REALS,
            )
        )

    def __rmod__(self, other: RandomVariable | Real) -> RandomVariable:

        if not is_known_subset(self._node.value_set, REALS):
            return NotImplemented

        if not isinstance(other, (RandomVariable, Real)):
            return NotImplemented

        if isinstance(other, RandomVariable):
            if not is_known_subset(other._node.value_set, REALS):
                return NotImplemented
        else:
            other = RandomVariable._from_node(
                ConstantNode(other)
            )

        return RandomVariable._from_node(
            OperationNode(
                operation=MODULO,
                inputs=(other._node,self._node),
                value_set=REALS,
            )
        )

    def _inequality_comparison(self, operator: Callable, other: RandomVariable | Real) -> Event:

        if not is_known_subset(self._node.value_set, REALS):
            return NotImplemented

        if isinstance(other, Real):
            other_node = ConstantNode(other)

        elif isinstance(other, RandomVariable):
            if not is_known_subset(other._node.value_set, REALS):
                return NotImplemented

            other_node = other._node

        else:
            return NotImplemented

        return Event(
            OperationNode(
                operation=operator,
                inputs=(self._node, other_node),
                value_set=sp.FiniteSet(False, True),
            )
        )

    def _equality_comparison(self, operator: Callable, other: RandomVariable | Real) -> Event:
        if isinstance(other, RandomVariable):
            other_node = other._node
        else:
            other_node = ConstantNode(other)

        return Event(
            OperationNode(
                operation=operator,
                inputs=(self._node, other_node),
                value_set=sp.FiniteSet(False, True),
            )
        )

    def __lt__(self, other: RandomVariable | Real) -> Event:
        return self._inequality_comparison(LT, other)

    def __le__(self, other: RandomVariable | Real) -> Event:
        return self._inequality_comparison(LTE, other)

    def __gt__(self, other: RandomVariable | Real) -> Event:
        return self._inequality_comparison(GT, other)

    def __ge__(self, other: RandomVariable | Real) -> Event:
        return self._inequality_comparison(GTE, other)

    def __eq__(self, other: RandomVariable | Real) -> Event:
        return self._equality_comparison(EQ, other)

    def __ne__(self, other: RandomVariable | Real) -> Event:
        return self._equality_comparison(NEQ, other)

    def __repr__(self):
        return (
            f"RandomVariable("
            f"name={self._name!r}, "
            f"value_set={self._node.value_set!r}"
            f")"
        )

    def __str__(self):
        return self.name

    def quantile_confidence_interval(self,
                                     q: float,
                                     alpha: float = DEF_ALPHA,
                                     num_samples: int = DEF_NUM_SAMPLES,
                                     rng: np.random.Generator | None = None
                                     ) -> ConfidenceInterval:

        samples = self.sample(
            num_samples=num_samples,
            rng=rng,
        )

        return quantile_confidence_interval(
            samples=samples,
            q=q,
            alpha=alpha,
        )




