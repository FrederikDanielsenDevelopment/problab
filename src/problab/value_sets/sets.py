import sympy as sp

from src.problab.value_sets.base import _UnknownValueSet

n = sp.Symbol("n", integer=True)

UNKNOWN_VALUE_SET        = _UnknownValueSet()
REALS                    = sp.S.Reals
POSITIVE_REALS           = sp.Interval.open(0, sp.oo)
NEGATIVE_REALS           = sp.Interval.open(-sp.oo, 0)
NON_NEGATIVE_REALS       = sp.Interval(0, sp.oo)
NON_POSITIVE_REALS       = sp.Interval(-sp.oo, 0)
INTEGERS                 = sp.S.Integers
POSITIVE_INTEGERS        = sp.S.Naturals
NEGATIVE_INTEGERS        = sp.Intersection(sp.S.Integers, sp.Interval.open(-sp.oo, 0))
NATURALS                 = sp.S.Naturals
NATURALS_0               = sp.S.Naturals0
COMPLEXES                = sp.S.Complexes
UNIT_INTERVAL            = sp.Interval(0, 1)
ZERO                     = sp.FiniteSet(0)
ONE                      = sp.FiniteSet(1)
NON_ZERO_REALS           = REALS - ZERO
NON_ZERO_COMPLEXES       = COMPLEXES - ZERO
NON_INTEGER_REALS        = REALS - INTEGERS
EVEN_INTEGERS            = sp.ImageSet(sp.Lambda(n, 2 * n), INTEGERS)
ODD_INTEGERS             = sp.ImageSet(sp.Lambda(n, 2 * n + 1), INTEGERS)
POSITIVE_EVEN_INTEGERS   = sp.Intersection(EVEN_INTEGERS, POSITIVE_INTEGERS)
NEGATIVE_EVEN_INTEGERS   = sp.Intersection(EVEN_INTEGERS, NEGATIVE_INTEGERS)
POSITIVE_ODD_INTEGERS    = sp.Intersection(ODD_INTEGERS, POSITIVE_INTEGERS)
NEGATIVE_ODD_INTEGERS    = sp.Intersection(ODD_INTEGERS, NEGATIVE_INTEGERS)
NON_POSITIVE_INTEGERS    = sp.Intersection(INTEGERS, NON_POSITIVE_REALS)