from __future__ import annotations
from types import NotImplementedType
import sympy as sp
from collections.abc import Iterable
from numbers import Real
from src.problab.distributions.base import Distribution



class RandomVariableSymbol(sp.Dummy):
    def __new__(cls, rv: RandomVariable) -> RandomVariableSymbol:
        symbol = super().__new__(cls)
        symbol.__random_variable = rv
        return symbol

    @property
    def random_variable(self) -> RandomVariable:
        return self.__random_variable

class RandomVariable:

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

    def __init__(self, distribution: Distribution = None, expression: sp.Expr = None):

        if distribution is None and expression is None:
            raise ValueError("A distribution or expression must be provided.")

        if distribution is not None and expression is not None:
            raise ValueError("Provide either a distribution or an expression, not both.")

        self.distribution = distribution

        self.symbol: RandomVariableSymbol | None

        if distribution is not None:
            self.symbol = RandomVariableSymbol(self)
            self.expression: sp.Expr = self.symbol
        else:
            self.symbol = None
            self.expression = expression

    def __add__(self, other: object) -> RandomVariable | NotImplementedType:
        other_expression = self._to_expression(other)
        if other_expression is NotImplemented: return NotImplemented
        return RandomVariable(expression=self.expression + other_expression)

    def __radd__(self, other: object) -> RandomVariable | NotImplementedType:
        return self.__add__(other)

    def __sub__(self, other: object) -> RandomVariable | NotImplementedType:
        other_expression = self._to_expression(other)
        if other_expression is NotImplemented: return NotImplemented
        return RandomVariable(expression=self.expression - other_expression)

    def __rsub__(self, other: object) -> RandomVariable | NotImplementedType:
        other_expression = self._to_expression(other)
        if other_expression is NotImplemented: return NotImplemented
        return RandomVariable(expression=other_expression - self.expression)

    def __mul__(self, other: object) -> RandomVariable | NotImplementedType:
        other_expression = self._to_expression(other)
        if other_expression is NotImplemented: return NotImplemented
        return RandomVariable(expression=self.expression * other_expression)

    def __rmul__(self, other: object) -> RandomVariable | NotImplementedType:
        return self.__mul__(other)

    def __truediv__(self, other: object) -> RandomVariable | NotImplementedType:
        other_expression = self._to_expression(other)
        if other_expression is NotImplemented: return NotImplemented
        return RandomVariable(expression=self.expression / other_expression)

    def __rtruediv__(self, other: object) -> RandomVariable | NotImplementedType:
        other_expression = self._to_expression(other)
        if other_expression is NotImplemented: return NotImplemented
        return RandomVariable(expression=other_expression / self.expression)

    def __floordiv__(self, other: object) -> RandomVariable | NotImplementedType:
        other_expression = self._to_expression(other)
        if other_expression is NotImplemented: return NotImplemented
        return RandomVariable(expression=self.expression // other_expression)

    def __rfloordiv__(self, other: object) -> RandomVariable | NotImplementedType:
        other_expression = self._to_expression(other)
        if other_expression is NotImplemented: return NotImplemented
        return RandomVariable(expression=other_expression // self.expression)

    def __mod__(self, other: object) -> RandomVariable | NotImplementedType:
        other_expression = self._to_expression(other)
        if other_expression is NotImplemented: return NotImplemented
        return RandomVariable(expression=self.expression % other_expression)

    def __rmod__(self, other: object) -> RandomVariable | NotImplementedType:
        other_expression = self._to_expression(other)
        if other_expression is NotImplemented: return NotImplemented
        return RandomVariable(expression=other_expression % self.expression)

    def __pow__(self, exponent: object, modulo: object | None = None) -> RandomVariable | NotImplementedType:
        if modulo is not None: return NotImplemented
        exponent_expression = self._to_expression(exponent)
        if exponent_expression is NotImplemented: return NotImplemented
        return RandomVariable(expression=self.expression ** exponent_expression)

    def __rpow__(self, base: object) -> RandomVariable | NotImplementedType:
        base_expression = self._to_expression(base)
        if base_expression is NotImplemented: return NotImplemented
        return RandomVariable(expression=base_expression ** self.expression)

    def __neg__(self) -> RandomVariable:
        return RandomVariable(expression=-self.expression)

    def __pos__(self) -> RandomVariable:
        return RandomVariable(expression=+self.expression)

    def __abs__(self) -> RandomVariable:
        return RandomVariable(expression=abs(self.expression))

    def __divmod__(self, other: object) -> tuple[RandomVariable, RandomVariable] | NotImplementedType:
        other_expression = self._to_expression(other)
        if other_expression is NotImplemented: return NotImplemented

        quotient_expression, remainder_expression = divmod(self.expression, other_expression)

        return (
            RandomVariable(expression=quotient_expression),
            RandomVariable(expression=remainder_expression),
        )


    def __rdivmod__(self, other: object) -> tuple[RandomVariable, RandomVariable] | NotImplementedType:
        ...

    def __round__(self, ndigits: int | None = None) -> RandomVariable:
        ...


class RandomVector:
    def __init__(self, components: Iterable[RandomVariable | Real]):
        for component in components:
            if not isinstance(component, Real):

        self.components = tuple(components)

    def __repr__(self) -> str:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[Real]:
        ...

    def __getitem__(self, index: int) -> Real:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __add__(self, other: object) -> Vector | NotImplementedType:
        ...

    def __radd__(self, other: object) -> Vector | NotImplementedType:
        ...

    def __sub__(self, other: object) -> Vector | NotImplementedType:
        ...

    def __rsub__(self, other: object) -> Vector | NotImplementedType:
        ...

    def __mul__(self, scalar: object) -> Vector | NotImplementedType:
        ...

    def __rmul__(self, scalar: object) -> Vector | NotImplementedType:
        ...

    def __truediv__(self, scalar: object) -> Vector | NotImplementedType:
        ...

    def __matmul__(self, other: object) -> Real | NotImplementedType:
        ...

    def __neg__(self) -> Vector:
        ...

    def __pos__(self) -> Vector:
        ...

    def __abs__(self) -> Real:
        ...

    def dot(self, other: Vector) -> Real:
        ...

    def cross(self, other: Vector) -> Vector:
        ...

    def norm(self) -> Real:
        ...

    def norm_squared(self) -> Real:
        ...

    def normalized(self) -> Vector:
        ...

    def distance_to(self, other: Vector) -> Real:
        ...

    def angle_to(self, other: Vector) -> Real:
        ...

    def project_onto(self, other: Vector) -> Vector:
        ...

