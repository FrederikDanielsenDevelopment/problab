from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Callable, Any

import numpy as np
import sympy as sp

from src.problab.distributions.base import Distribution
from src.problab.random_variables.context import RealizationContext
from src.problab.value_sets.base import ValueSet
from src.problab.value_sets.sets import UNKNOWN_VALUE_SET

T = TypeVar('T')


class Node(ABC, Generic[T]):

    def __init__(self):
        self._name = None

    @property
    @abstractmethod
    def value_set(self) -> ValueSet:
        ...

    @property
    @abstractmethod
    def dependencies(self) -> set[Node[Any]]:
        # Only top level of dependencies not a graph of dependencies of dependencies.
        ...

    @abstractmethod
    def _evaluate(self, context: RealizationContext) -> np.ndarray:
        ...

    @property
    def name(self) -> str:
        return self._name

    @property
    def has_dependencies(self) -> bool:
        return bool(self.dependencies)

class ConstantNode(Node[T]):

    def __init__(self, value: T) -> None:
        super().__init__()

        self._value = value
        self._name = str(value)

    @property
    def value(self) -> T:
        return self._value

    @property
    def value_set(self) -> ValueSet:
        return sp.FiniteSet(self._value)

    @property
    def dependencies(self) -> set[Node[Any]]:
        return set()

    def _evaluate(self, context: RealizationContext) -> np.ndarray:
        return np.asarray(self._value)


class DistributionNode(Node[T]):

    def __init__(self, distribution: Distribution[T], rv_name: str) -> None:
        super().__init__()

        self._distribution = distribution
        self._name = f"{rv_name} ~ {distribution.name}"

    @property
    def distribution(self) -> Distribution[T]:
        return self._distribution

    @property
    def value_set(self) -> ValueSet:
        return self._distribution.value_set

    @property
    def dependencies(self) -> set[Node[Any]]:
        return self._distribution.node_dependencies

    def _evaluate(self, context: RealizationContext) -> np.ndarray:
        return self._distribution.sample(context)


class OperationNode(Node[T]):

    def __init__(self,
                 operation: Callable[..., T],
                 inputs: tuple[Node[Any], ...],
                 name: str,
                 value_set: ValueSet = UNKNOWN_VALUE_SET,
        ) -> None:
        super().__init__()

        self._operation = operation
        self._inputs = inputs
        self._value_set = value_set
        self._name = name

    @property
    def value_set(self) -> ValueSet:
        return self._value_set

    @property
    def dependencies(self) -> set[Node[Any]]:
        return set(self._inputs)

    def _evaluate(self, context: RealizationContext) -> np.ndarray:
        values = tuple(
            context.evaluate(node)
            for node in self._inputs
        )

        return self._operation(*values)





