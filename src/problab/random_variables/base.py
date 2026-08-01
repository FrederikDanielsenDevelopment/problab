from __future__ import annotations

from abc import abstractmethod, ABC
from types import NotImplementedType
from typing import Iterator
import sympy as sp
from numbers import Real
from src.problab.distributions.base import Distribution, DegenerateDistribution
import numpy as np
from collections.abc import Callable
from sympy.utilities.lambdify import implemented_function
from itertools import count
from operator import index

_ROUND_FUNCTION = implemented_function(
    sp.Function("round"),
    np.round,
)

class RandomVariableSymbol(sp.Dummy):

    def __new__(cls, rv: _SourceRandomVariable, name: str | None = None) -> RandomVariableSymbol:
        symbol = super().__new__(cls, name)
        symbol.__random_variable = rv
        return symbol

    @property
    def random_variable(self) -> _SourceRandomVariable:
        return self.__random_variable


class RandomVariable(ABC):

    _count = count()
    _applied_function_count = count()

    @staticmethod
    def _from_expression(expression: sp.Expr, name: str | None = None) -> RandomVariable:
        return _DerivedRandomVariable(expression=expression, name=name)


    @staticmethod
    def _to_expression(value: object) -> sp.Expr | NotImplementedType:
        if isinstance(value, RandomVariable):
            return value.expression

        try:
            expression = sp.sympify(value, strict=True)
        except (sp.SympifyError, TypeError):
            return NotImplemented

        if not isinstance(expression, sp.Expr):
            return NotImplemented

        if expression.is_number is not True:
            return NotImplemented

        if expression.is_real is not True:
            return NotImplemented

        return expression

    def __new__(cls,
                *args: object,
                **kwargs: object,) -> RandomVariable:
        if cls is RandomVariable:
            return object.__new__(_SourceRandomVariable)

        return object.__new__(cls)

    def __init__(self, name: str | None = None) -> None:
        self._name = (
            f"RV_{next(RandomVariable._count)}"
            if name is None
            else name
        )

    @property
    @abstractmethod
    def expression(self) -> sp.Expr:
        ...

    @abstractmethod
    def realize(self) -> Real:
        ...

    @abstractmethod
    def sample(self, num_samples: int | None = None) -> np.ndarray:
        ...

    @property
    def name(self) -> str:
        return self._name

    def apply(self, function: Callable[[Real], Real], name: str = None) -> RandomVariable:
        f = implemented_function(
            sp.Function(name if name is not None else f"f_{next(self._applied_function_count)}"),
            function
        )

        return _DerivedRandomVariable(expression=f(self.expression))

    def __repr__(self) -> str:
        return repr(self.expression)

    def __str__(self) -> str:
        return str(self.expression)

    def __add__(self, other: object) -> RandomVariable | NotImplementedType:
        if other_expression := self._to_expression(other) is NotImplemented: return NotImplemented
        return self._from_expression(self.expression + other_expression)

    def __radd__(self, other: object) -> RandomVariable | NotImplementedType:
        return self.__add__(other)

    def __sub__(self, other: object) -> RandomVariable | NotImplementedType:
        if other_expression := self._to_expression(other) is NotImplemented: return NotImplemented
        return self._from_expression(self.expression - other_expression)

    def __rsub__(self, other: object) -> RandomVariable | NotImplementedType:
        if other_expression := self._to_expression(other) is NotImplemented: return NotImplemented
        return self._from_expression(other_expression - self.expression)

    def __mul__(self, other: object) -> RandomVariable | NotImplementedType:
        if other_expression := self._to_expression(other) is NotImplemented: return NotImplemented
        return self._from_expression(self.expression * other_expression)

    def __rmul__(self, other: object) -> RandomVariable | NotImplementedType:
        return self.__mul__(other)

    def __truediv__(self, other: object) -> RandomVariable | NotImplementedType:
        if other_expression := self._to_expression(other) is NotImplemented: return NotImplemented
        return self._from_expression(self.expression / other_expression)

    def __rtruediv__(self, other: object) -> RandomVariable | NotImplementedType:
        if other_expression := self._to_expression(other) is NotImplemented: return NotImplemented
        return self._from_expression(other_expression / self.expression)

    def __floordiv__(self, other: object) -> RandomVariable | NotImplementedType:
        if other_expression := self._to_expression(other) is NotImplemented: return NotImplemented
        return self._from_expression(self.expression // other_expression)

    def __rfloordiv__(self, other: object) -> RandomVariable | NotImplementedType:
        if other_expression := self._to_expression(other) is NotImplemented: return NotImplemented
        return self._from_expression(other_expression // self.expression)

    def __mod__(self, other: object) -> RandomVariable | NotImplementedType:
        if other_expression := self._to_expression(other) is NotImplemented: return NotImplemented
        return self._from_expression(self.expression % other_expression)

    def __rmod__(self, other: object) -> RandomVariable | NotImplementedType:
        if other_expression := self._to_expression(other) is NotImplemented: return NotImplemented
        return self._from_expression(other_expression % self.expression)

    def __pow__(self, exponent: object, modulo: object | None = None) -> RandomVariable | NotImplementedType:
        if modulo is not None: return NotImplemented
        if exponent_expression := self._to_expression(exponent) is NotImplemented: return NotImplemented
        return _DerivedRandomVariable(expression=self.expression ** exponent_expression)

    def __rpow__(self, base: object) -> RandomVariable | NotImplementedType:
        if base_expression := self._to_expression(base) is NotImplemented: return NotImplemented
        return self._from_expression(base_expression ** self.expression)

    def __neg__(self) -> RandomVariable:
        return self._from_expression(sp.simplify(-self.expression))

    def __pos__(self) -> RandomVariable:
        return self._from_expression(+self.expression)

    def __abs__(self) -> RandomVariable:
        return self._from_expression(abs(self.expression))

    def __divmod__(self, other: object) -> tuple[RandomVariable, RandomVariable] | NotImplementedType:
        if other_expression := self._to_expression(other) is NotImplemented: return NotImplemented

        quotient_expression, remainder_expression = divmod(self.expression, other_expression)

        return (
            _DerivedRandomVariable(expression=quotient_expression),
            _DerivedRandomVariable(expression=remainder_expression),
        )

    def __rdivmod__(self, other: object) -> tuple[RandomVariable, RandomVariable] | NotImplementedType:
        if other_expression := self._to_expression(other) is NotImplemented: return NotImplemented

        quotient_expression, remainder_expression = divmod(other_expression, self.expression)

        return (
            _DerivedRandomVariable(expression=quotient_expression),
            _DerivedRandomVariable(expression=remainder_expression),
        )

    def __round__(self, n_digits: int | None = None) -> RandomVariable:
        if n_digits is None:
            digits = 0
        else:
            try:
                digits = index(n_digits)
            except TypeError:
                raise TypeError("n_digits must be an integer or None.") from None

        return self._from_expression(
            _ROUND_FUNCTION(
                self.expression,
                digits,
            )
        )


class _SourceRandomVariable(RandomVariable):

    def __init__(self,
                 distribution: Distribution,
                 name: str | None = None):

        if not isinstance(distribution, Distribution):
            raise TypeError("A distribution must be provided.")

        super().__init__(name)

        self._distribution: Distribution = distribution
        self._symbol: RandomVariableSymbol = RandomVariableSymbol(self, name=self._name)

    @property
    def distribution(self) -> Distribution:
        return self._distribution

    @property
    def expression(self) -> RandomVariableSymbol:
        return self._symbol

    def realize(self) -> Real:
        return self.distribution.sample()[0]

    def sample(self, n_samples: int | None = None) -> np.ndarray:
        return self.distribution.sample(n_samples)


class _DerivedRandomVariable(RandomVariable):

    def __init__(self,
                 expression: sp.Expr,
                 name: str | None = None):

        if not isinstance(expression, sp.Expr):
            raise TypeError("An expression must be provided.")

        super().__init__(name)

        self._expression = expression

    @property
    def expression(self) -> sp.Expr:
        return self._expression

    def realize(self) -> Real:

        realizations: dict[RandomVariableSymbol, Real] = {}

        for symbol in self.expression.free_symbols:
            realizations[symbol] = symbol.random_variable.realize()

        return self.expression.subs(realizations)

    def sample(self, n_samples: int | None = None) -> np.ndarray:
        if n_samples is None: return np.array([self.realize()])
        return np.array([self.realize() for _ in range(n_samples)])


class RandomArray:

    @staticmethod
    def _to_array_operand(other: object,) -> np.ndarray | NotImplementedType:
        if isinstance(other, RandomArray): return other.__array
        if isinstance(other, np.ndarray): return other
        return NotImplemented

    def __init__(self, components: object, *, copy: bool | None = None, ndmin: int = 0) -> None:

        self.__array: np.ndarray

        try:
            self.__array = np.array(components, dtype=object, copy=copy, ndmin=ndmin)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "'components' must be convertible to a NumPy array"
            ) from exc

        for idx, component in np.ndenumerate(self.__array):
            if isinstance(component, Real):
                self.__array[idx] = RandomVariable(
                    distribution=DegenerateDistribution(value=component)
                )
            elif isinstance(component, RandomVariable):
                continue
            else:
                raise TypeError(
                    f"Invalid element at index {idx}: "
                    f"expected RandomVariable or Real, "
                    f"got {type(component).__name__}"
                )

    __hash__ = None

    def __repr__(self) -> str:
        return "[" + ", ".join([repr(RV) for RV in self.__array]) + "]"

    def __str__(self) -> str:
        return "[" + ", ".join([str(RV) for RV in self.__array]) + "]"

    def __len__(self) -> int:
        return len(self.__array)

    def __iter__(self) -> Iterator[RandomVariable |np.ndarray]:
        return iter(self.__array)

    def __getitem__(self, index: int) -> Real:
        return self.__array[index]

    def __eq__(self, other: object) -> bool | NotImplementedType:

        if not isinstance(other, RandomArray):
            return NotImplemented

        if self.__array.shape != other.__array.shape:
            return False

        return all(
            component1 == component2
            for component1, component2 in zip(
                self.__array.flat,
                other.__array.flat,
            )
        )

    def __add__(self, other: object) -> RandomArray | NotImplementedType:

        other_array = self._to_array_operand(other)

        if other_array is NotImplemented: return NotImplemented

        if self.__array.shape != other_array.shape:
            raise ValueError(
                f"Incompatible shapes: "
                f"{self.__array.shape} and {other_array.shape}"
            )

        return RandomArray(components=self.__array + other_array)

    def __radd__(self, other: object) -> RandomArray | NotImplementedType:
        return self.__add__(other)

    def __sub__(self, other: object) -> RandomArray | NotImplementedType:

        other_array = self._to_array_operand(other)

        if other_array is NotImplemented: return NotImplemented

        if self.__array.shape != other_array.shape:
            raise ValueError(
                f"Incompatible shapes: "
                f"{self.__array.shape} and {other_array.shape}"
            )

        return RandomArray(components=self.__array - other_array)

    def __rsub__(self, other: object) -> RandomArray | NotImplementedType:

        other_array = self._to_array_operand(other)

        if other_array is NotImplemented: return NotImplemented

        if self.__array.shape != other_array.shape:
            raise ValueError(
                f"Incompatible shapes: "
                f"{self.__array.shape} and {other_array.shape}"
            )

        return RandomArray(components=other_array - self.__array)

    def __mul__(self, other: object) -> RandomArray | NotImplementedType:

        other_array = self._to_array_operand(other)

        if other_array is NotImplemented: return NotImplemented

        if self.__array.shape != other_array.shape:
            raise ValueError(
                f"Incompatible shapes: "
                f"{self.__array.shape} and {other_array.shape}"
            )

        return RandomArray(components=self.__array * other_array)

    def __rmul__(self, other: object) -> RandomArray | NotImplementedType:
        return self.__mul__(other)

    # Implements division of every component by a scalar or compatible operand.
    def __truediv__(self, scalar: object) -> RandomArray | NotImplementedType:
        ...

    # Implements the matrix-multiplication operator, typically as a dot product or matrix product.
    def __matmul__(self, other: object) -> Real | NotImplementedType:
        ...

    # Returns a new RandomArray with every component negated.
    def __neg__(self) -> RandomArray:
        ...

    # Returns the RandomArray unchanged or as an equivalent positive copy.
    def __pos__(self) -> RandomArray:
        ...

    # Returns a new RandomArray containing the absolute value of every component.
    def __abs__(self) -> RandomArray:
        new_array = np.empty_like(self.__array, dtype=object)

        for idx, component in np.ndenumerate(self.__array):
            new_array[idx] = abs(component)

        return RandomArray(components=new_array)

    # Calculates the dot product between this array and another vector.
    def dot(self, other: RandomArray) -> Real:
        ...

    # Calculates the three-dimensional cross product between this array and another vector.
    def cross(self, other: RandomArray) -> RandomArray:
        ...

    # Calculates the Euclidean length of the vector.
    def norm(self) -> Real:
        ...

    # Calculates the square of the Euclidean length without taking a square root.
    def norm_squared(self) -> Real:
        ...

    # Returns a vector with the same direction and a norm of one.
    def normalized(self) -> RandomArray:
        ...

    # Calculates the Euclidean distance between this vector and another vector.
    def distance_to(self, other: RandomArray) -> Real:
        ...

    # Calculates the angle between this vector and another vector.
    def angle_to(self, other: RandomArray) -> Real:
        ...

    # Returns the projection of this vector onto another vector.
    def project_onto(self, other: RandomArray) -> RandomArray:
        ...

