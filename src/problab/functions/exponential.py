import numpy as np
import sympy as sp

from numbers import Real

from src.problab.random_variables.base import RandomVariable
from src.problab.functions._utils import (
    _apply_scalar_or_rv,
    _require_domain,
    _require_real_valued,
)


def exp(x: RandomVariable | Real) -> RandomVariable | Real:
    _require_real_valued(x)

    return _apply_scalar_or_rv(
        x=x,
        function=np.exp,
        value_set=sp.Interval.open(0, sp.oo),
    )


def log(x: RandomVariable | Real) -> RandomVariable | Real:
    _require_domain(
        x=x,
        domain=sp.Interval.open(0, sp.oo),
    )

    return _apply_scalar_or_rv(
        x=x,
        function=np.log,
        value_set=sp.S.Reals,
    )


def log2(x: RandomVariable | Real) -> RandomVariable | Real:
    _require_domain(
        x=x,
        domain=sp.Interval.open(0, sp.oo),
    )

    return _apply_scalar_or_rv(
        x=x,
        function=np.log2,
        value_set=sp.S.Reals,
    )


def log10(x: RandomVariable | Real) -> RandomVariable | Real:
    _require_domain(
        x=x,
        domain=sp.Interval.open(0, sp.oo),
    )

    return _apply_scalar_or_rv(
        x=x,
        function=np.log10,
        value_set=sp.S.Reals,
    )