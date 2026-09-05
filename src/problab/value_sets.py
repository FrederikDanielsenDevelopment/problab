
from typing import TypeAlias

import sympy as sp

class _UnknownValueSet:

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNKNOWN_VALUE_SET"

ValueSet: TypeAlias = sp.Set | _UnknownValueSet

UNKNOWN_VALUE_SET = _UnknownValueSet()
REALS = sp.S.Reals
POSITIVE_REALS = sp.Interval.open(0, sp.oo)
NEGATIVE_REALS = sp.Interval.open(-sp.oo, 0)
NON_NEGATIVE_REALS = sp.Interval(0, sp.oo)
NON_POSITIVE_REALS = sp.Interval(-sp.oo, 0)
INTEGERS = sp.S.Integers
POSITIVE_INTEGERS = sp.S.PositiveIntegers
NEGATIVE_INTEGERS = sp.S.NegativeIntegers
NATURALS = sp.S.Naturals
NATURALS_0 = sp.S.Naturals0
COMPLEXES = sp.S.Complexes
UNIT_INTERVAL = sp.Interval(0, 1)


def is_known_subset(subset: ValueSet, superset: sp.Set) -> bool:
    if subset is UNKNOWN_VALUE_SET:
        return False

    return subset.is_subset(superset) is True