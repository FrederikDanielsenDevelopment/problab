from __future__ import annotations
from types import NotImplementedType
import sympy as sp



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

        if expression is None:
            self.expression = self.symbol

    def __add__(self, other: object) -> RandomVariable | NotImplementedType:
        other_expression = self._to_expression(other)
        if other_expression is NotImplemented:
            return NotImplemented
        return RandomVariable(expression=self.expression + other_expression)

    def __radd__(self, other: object) -> RandomVariable | NotImplementedType:
        return self.__add__(other)

    def __sub__(self, other: object) -> RandomVariable | NotImplementedType:
        other_expression = self._to_expression(other)
        if other_expression is NotImplemented:
            return NotImplemented
        return RandomVariable(expression=self.expression - other_expression)

    def __rsub__(self, other: object) -> RandomVariable | NotImplementedType:
        other_expression = self._to_expression(other)
        if other_expression is NotImplemented:
            return NotImplemented
        return RandomVariable(expression=other_expression - self.expression)

    def __mul__(self, other: object) -> RandomVariable | NotImplementedType:
        other_expression = self._to_expression(other)
        if other_expression is NotImplemented:
            return NotImplemented
        return RandomVariable(expression=self.expression * other_expression)

    def __rmul__(self, other: object) -> RandomVariable | NotImplementedType:
        return self.__mul__(other)

    def __truediv__(self, other: object) -> RandomVariable | NotImplementedType:
        other_expression = self._to_expression(other)
        if other_expression is NotImplemented:
            return NotImplemented
        return RandomVariable(expression=self.expression / other_expression)

    def __rtruediv__(self, other: object) -> RandomVariable | NotImplementedType:
        other_expression = self._to_expression(other)
        if other_expression is NotImplemented:
            return NotImplemented
        return RandomVariable(expression=other_expression / self.expression)

    def __floordiv__(self, other: object) -> RandomVariable | NotImplementedType:
        other_expression = self._to_expression(other)
        if other_expression is NotImplemented:
            return NotImplemented
        return RandomVariable(expression=other_expression // self.expression)

    def __rfloordiv__(self, other: object) -> RandomVariable | NotImplementedType:
        other_expression = self._to_expression(other)
        if other_expression is NotImplemented:
            return NotImplemented
        return RandomVariable(expression=self.expression // other_expression)

    def __mod__(self, other: object) -> RandomVariable | NotImplementedType:
        other_expression = self._to_expression(other)
        if other_expression is NotImplemented:
            return NotImplemented
        return RandomVariable(expression=other_expression % self.expression)

    def __rmod__(self, other: object) -> RandomVariable | NotImplementedType:
        other_expression = self._to_expression(other)
        if other_expression is NotImplemented:
            return NotImplemented
        return RandomVariable(expression=self.expression % other_expression)

    def __pow__(self, exponent: object, modulo: object | None = None) -> RandomVariable | NotImplementedType:
        # should we have checks that the exponent is in a set of supported types here?
        return RandomVariable(expression=self.expression ** exponent)

    def __rpow__(self, base: object) -> RandomVariable | NotImplementedType:
        ...

    def __neg__(self) -> RandomVariable:
        ...

    def __pos__(self) -> RandomVariable:
        ...

    def __abs__(self) -> RandomVariable:
        ...

    def __divmod__(self, other: object) -> tuple[RandomVariable, RandomVariable] | NotImplementedType:
        ...

    def __rdivmod__(self, other: object) -> tuple[RandomVariable, RandomVariable] | NotImplementedType:
        ...

    def __round__(self, ndigits: int | None = None) -> RandomVariable:
        ...


class Expression:
    def __init__(self, expression: sp.Expr):
        self.expression = expression

    def __add__(self, other: Expression):

        symbol_mapping: dict[sp.Symbol, sp.Symbol] = {}

        # rename overlapping symbols so they align name-wise
        for symbol1, RV1 in self.dependencies.items():
            for symbol2, RV2 in other.dependencies.items():
                if (RV1 is RV2) and (symbol1 != symbol2):
                    symbol_mapping[symbol2] = symbol1

        # rename symbols in other list tied to dependencies that are not present in self dependencies
        new_symbol_idx = len(self.dependencies) + 1
        for symbol, RV in other.dependencies.items():
            if not any(RV is existing_rv for existing_rv in self.dependencies.values()):
                new_symbol = sp.Symbol("x" + str(new_symbol_idx))
                new_symbol_idx += 1
                symbol_mapping[symbol] = new_symbol

        other_expression_aligned = other.expression.xreplace(symbol_mapping)

        other_dependencies_aligned = {
            symbol_mapping.get(old_symbol, old_symbol): RV
            for old_symbol, RV in other.dependencies.items()
        }


        combined_dependencies = (self.dependencies | other_dependencies_aligned)

        combined_expression: Expression = Expression(expression=self.expression + other_expression_aligned,
                                                     dependencies=combined_dependencies)

        return combined_expression

def