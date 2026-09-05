import numpy as np
import sympy as sp

from numbers import Real

from src.problab.random_variables.base import RandomVariable
from src.problab.functions._utils import (
    _apply_scalar_or_rv,
    _require_domain,
    _require_real_valued,
)


def sqrt(x: RandomVariable | Real) -> RandomVariable | Real:
    _require_domain(
        x=x,
        domain=sp.Interval(0, sp.oo),
    )

    return _apply_scalar_or_rv(
        x=x,
        function=np.sqrt,
        value_set=sp.Interval(0, sp.oo),
    )


def absolute(x: RandomVariable | Real) -> RandomVariable | Real:
    _require_real_valued(x)

    return _apply_scalar_or_rv(
        x=x,
        function=np.abs,
        value_set=sp.Interval(0, sp.oo),
    )


def floor(x: RandomVariable | Real) -> RandomVariable | Real:
    _require_real_valued(x)

    return _apply_scalar_or_rv(
        x=x,
        function=np.floor,
        value_set=sp.S.Integers,
    )


def ceil(x: RandomVariable | Real) -> RandomVariable | Real:
    _require_real_valued(x)

    return _apply_scalar_or_rv(
        x=x,
        function=np.ceil,
        value_set=sp.S.Integers,
    )


def sign(x: RandomVariable | Real) -> RandomVariable | Real:
    _require_real_valued(x)

    return _apply_scalar_or_rv(
        x=x,
        function=np.sign,
        value_set=sp.FiniteSet(-1, 0, 1),
    )