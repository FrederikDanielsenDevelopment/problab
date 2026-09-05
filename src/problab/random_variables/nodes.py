from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Callable, Any

import numpy as np

from src.problab.distributions.base import Distribution
from src.problab.random_variables.context import RealizationContext
from src.problab.value_sets import ValueSet, UNKNOWN_VALUE_SET

T = TypeVar('T')


class Node(ABC, Generic[T]):

    @property
    @abstractmethod
    def value_set(self) -> ValueSet:
        ...

    @abstractmethod
    def _evaluate(self, context: RealizationContext) -> np.ndarray:
        ...


class ConstantNode(Node[T]):

    def __init__(self, value: T) -> None:
        self._value = value

    @property
    def value(self) -> T:
        return self._value

    @property
    def value_set(self) -> ValueSet:
        return sp.FiniteSet(self._value)

    def _evaluate(self, context: RealizationContext) -> np.ndarray:
        return np.asarray(self._value)


class DistributionNode(Node[T]):

    def __init__(self, distribution: Distribution[T]) -> None:
        self._distribution = distribution

    @property
    def distribution(self) -> Distribution[T]:
        return self._distribution

    @property
    def value_set(self) -> ValueSet:
        return self._distribution.value_set

    def _evaluate(self, context: RealizationContext) -> np.ndarray:
        return self._distribution.sample(context)


class OperationNode(Node[T]):

    def __init__(self,
                 operation: Callable[..., T],
                 inputs: tuple[Node[Any], ...],
                 value_set: ValueSet = UNKNOWN_VALUE_SET,
        ) -> None:
        self._operation = operation
        self._inputs = inputs
        self._value_set = value_set


    @property
    def value_set(self) -> ValueSet:
        return self._value_set

    def _evaluate(self, context: RealizationContext) -> np.ndarray:
        values = tuple(
            context.evaluate(node)
            for node in self._inputs
        )

        return self._operation(*values)





