from numbers import Real

import numpy as np
import sympy as sp


from src.problab.functions._utils import _require_real_valued, _apply, _require_domain
from src.problab.random_variables.base import RandomVariable

def sin(x: RandomVariable | Real) -> RandomVariable | float:
    _require_real_valued(x)

    return _apply(
        x=x,
        function=np.sin,
        value_set=sp.Interval(-1, 1),
    )


def cos(x: RandomVariable | Real) -> RandomVariable | float:
    _require_real_valued(x)

    return _apply(
        x=x,
        function=np.cos,
        value_set=sp.Interval(-1, 1),
    )


def tan(x: RandomVariable | Real) -> RandomVariable | float:
    _require_real_valued(x)

    return _apply(
        x=x,
        function=np.tan,
        value_set=sp.S.Reals,
    )


def arcsin(x: RandomVariable | Real) -> RandomVariable | float:
    _require_domain(
        x=x,
        domain=sp.Interval(-1, 1),
    )

    return _apply(
        x=x,
        function=np.arcsin,
        value_set=sp.Interval(-sp.pi / 2, sp.pi / 2),
    )


def arccos(x: RandomVariable | Real) -> RandomVariable | float:
    _require_domain(
        x=x,
        domain=sp.Interval(-1, 1),
    )

    return _apply(
        x=x,
        function=np.arccos,
        value_set=sp.Interval(0, sp.pi),
    )


def arctan(x: RandomVariable | Real) -> RandomVariable | float:
    _require_real_valued(x)

    return _apply(
        x=x,
        function=np.arctan,
        value_set=sp.Interval.open(-sp.pi / 2, sp.pi / 2),
    )


def sinh(x: RandomVariable | Real) -> RandomVariable | float:
    _require_real_valued(x)

    return _apply(
        x=x,
        function=np.sinh,
        value_set=sp.S.Reals,
    )


def cosh(x: RandomVariable | Real) -> RandomVariable | float:
    _require_real_valued(x)

    return _apply(
        x=x,
        function=np.cosh,
        value_set=sp.Interval(1, sp.oo),
    )


def tanh(x: RandomVariable | Real) -> RandomVariable | float:
    _require_real_valued(x)

    return _apply(
        x=x,
        function=np.tanh,
        value_set=sp.Interval.open(-1, 1),
    )


def arcsinh(x: RandomVariable | Real) -> RandomVariable | float:
    _require_real_valued(x)

    return _apply(
        x=x,
        function=np.arcsinh,
        value_set=sp.S.Reals,
    )


def arccosh(x: RandomVariable | Real) -> RandomVariable | float:
    _require_domain(
        x=x,
        domain=sp.Interval(1, sp.oo),
    )

    return _apply(
        x=x,
        function=np.arccosh,
        value_set=sp.Interval(0, sp.oo),
    )


def arctanh(x: RandomVariable | Real) -> RandomVariable | float:
    _require_domain(
        x=x,
        domain=sp.Interval.open(-1, 1),
    )

    return _apply(
        x=x,
        function=np.arctanh,
        value_set=sp.S.Reals,
    )