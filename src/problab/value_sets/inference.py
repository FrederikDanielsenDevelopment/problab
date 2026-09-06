import sympy as sp

from src.problab.value_sets._utils import is_known_subset
from src.problab.value_sets.base import ValueSet
from src.problab.value_sets.sets import UNKNOWN_VALUE_SET, ZERO, ONE, POSITIVE_REALS, REALS, NON_NEGATIVE_REALS, \
    POSITIVE_EVEN_INTEGERS, NON_ZERO_REALS, NEGATIVE_EVEN_INTEGERS, POSITIVE_ODD_INTEGERS, NEGATIVE_ODD_INTEGERS, \
    NEGATIVE_REALS, NON_INTEGER_REALS, COMPLEXES, NON_ZERO_COMPLEXES, NON_POSITIVE_REALS, INTEGERS, NATURALS_0, \
    NEGATIVE_INTEGERS, POSITIVE_INTEGERS, NON_POSITIVE_INTEGERS

n = sp.Symbol("n", integer=True)


# Binary operations

def _infer_add_value_set(left_set: ValueSet, right_set: ValueSet) -> ValueSet:

    if left_set == ZERO:
        return right_set

    if right_set == ZERO:
        return left_set

    if (
        is_known_subset(left_set, POSITIVE_REALS)
        and is_known_subset(right_set, POSITIVE_REALS)
    ):
        return POSITIVE_REALS

    if (
        is_known_subset(left_set, NON_NEGATIVE_REALS)
        and is_known_subset(right_set, NON_NEGATIVE_REALS)
    ):
        return NON_NEGATIVE_REALS

    if (
        is_known_subset(left_set, NEGATIVE_REALS)
        and is_known_subset(right_set, NEGATIVE_REALS)
    ):
        return NEGATIVE_REALS

    if (
        is_known_subset(left_set, NON_POSITIVE_REALS)
        and is_known_subset(right_set, NON_POSITIVE_REALS)
    ):
        return NON_POSITIVE_REALS

    if (
        is_known_subset(left_set, INTEGERS)
        and is_known_subset(right_set, INTEGERS)
    ):
        return INTEGERS

    if (
        is_known_subset(left_set, REALS)
        and is_known_subset(right_set, REALS)
    ):
        return REALS

    if (
        is_known_subset(left_set, COMPLEXES)
        and is_known_subset(right_set, COMPLEXES)
    ):
        return COMPLEXES

    return UNKNOWN_VALUE_SET


def _infer_subtract_value_set(left_set: ValueSet, right_set: ValueSet) -> ValueSet:

    if right_set == ZERO:
        return left_set

    if (
        is_known_subset(left_set, POSITIVE_REALS)
        and is_known_subset(right_set, NEGATIVE_REALS)
    ):
        return POSITIVE_REALS

    if (
        is_known_subset(left_set, NON_NEGATIVE_REALS)
        and is_known_subset(right_set, NON_POSITIVE_REALS)
    ):
        return NON_NEGATIVE_REALS

    if (
        is_known_subset(left_set, NEGATIVE_REALS)
        and is_known_subset(right_set, POSITIVE_REALS)
    ):
        return NEGATIVE_REALS

    if (
        is_known_subset(left_set, NON_POSITIVE_REALS)
        and is_known_subset(right_set, NON_NEGATIVE_REALS)
    ):
        return NON_POSITIVE_REALS

    if (
        is_known_subset(left_set, INTEGERS)
        and is_known_subset(right_set, INTEGERS)
    ):
        return INTEGERS

    if (
        is_known_subset(left_set, REALS)
        and is_known_subset(right_set, REALS)
    ):
        return REALS

    if (
        is_known_subset(left_set, COMPLEXES)
        and is_known_subset(right_set, COMPLEXES)
    ):
        return COMPLEXES

    return UNKNOWN_VALUE_SET


def _infer_multiply_value_set(left_set: ValueSet, right_set: ValueSet) -> ValueSet:

    if left_set == ZERO or right_set == ZERO:
        return ZERO

    if left_set == ONE:
        return right_set

    if right_set == ONE:
        return left_set

    if (
        is_known_subset(left_set, POSITIVE_REALS)
        and is_known_subset(right_set, POSITIVE_REALS)
    ):
        return POSITIVE_REALS

    if (
        is_known_subset(left_set, NEGATIVE_REALS)
        and is_known_subset(right_set, NEGATIVE_REALS)
    ):
        return POSITIVE_REALS

    if (
        is_known_subset(left_set, POSITIVE_REALS)
        and is_known_subset(right_set, NEGATIVE_REALS)
        or is_known_subset(left_set, NEGATIVE_REALS)
        and is_known_subset(right_set, POSITIVE_REALS)
    ):
        return NEGATIVE_REALS

    if (
        is_known_subset(left_set, NON_NEGATIVE_REALS)
        and is_known_subset(right_set, NON_NEGATIVE_REALS)
    ):
        return NON_NEGATIVE_REALS

    if (
        is_known_subset(left_set, NON_POSITIVE_REALS)
        and is_known_subset(right_set, NON_POSITIVE_REALS)
    ):
        return NON_NEGATIVE_REALS

    if (
        is_known_subset(left_set, INTEGERS)
        and is_known_subset(right_set, INTEGERS)
    ):
        return INTEGERS

    if (
        is_known_subset(left_set, NON_ZERO_REALS)
        and is_known_subset(right_set, NON_ZERO_REALS)
    ):
        return NON_ZERO_REALS

    if (
        is_known_subset(left_set, REALS)
        and is_known_subset(right_set, REALS)
    ):
        return REALS

    if (
        is_known_subset(left_set, NON_ZERO_COMPLEXES)
        and is_known_subset(right_set, NON_ZERO_COMPLEXES)
    ):
        return NON_ZERO_COMPLEXES

    if (
        is_known_subset(left_set, COMPLEXES)
        and is_known_subset(right_set, COMPLEXES)
    ):
        return COMPLEXES

    return UNKNOWN_VALUE_SET


def _infer_divide_value_set(numerator_set: ValueSet, denominator_set: ValueSet) -> ValueSet:

    if numerator_set == ZERO:
        return ZERO

    if denominator_set == ONE:
        return numerator_set

    if (
        is_known_subset(numerator_set, POSITIVE_REALS)
        and is_known_subset(denominator_set, POSITIVE_REALS)
    ):
        return POSITIVE_REALS

    if (
        is_known_subset(numerator_set, NEGATIVE_REALS)
        and is_known_subset(denominator_set, NEGATIVE_REALS)
    ):
        return POSITIVE_REALS

    if (
        is_known_subset(numerator_set, POSITIVE_REALS)
        and is_known_subset(denominator_set, NEGATIVE_REALS)
        or is_known_subset(numerator_set, NEGATIVE_REALS)
        and is_known_subset(denominator_set, POSITIVE_REALS)
    ):
        return NEGATIVE_REALS

    if (
        is_known_subset(numerator_set, NON_ZERO_REALS)
        and is_known_subset(denominator_set, NON_ZERO_REALS)
    ):
        return NON_ZERO_REALS

    if (
        is_known_subset(numerator_set, REALS)
        and is_known_subset(denominator_set, NON_ZERO_REALS)
    ):
        return REALS

    if (
        is_known_subset(numerator_set, NON_ZERO_COMPLEXES)
        and is_known_subset(denominator_set, NON_ZERO_COMPLEXES)
    ):
        return NON_ZERO_COMPLEXES

    if (
        is_known_subset(numerator_set, COMPLEXES)
        and is_known_subset(denominator_set, NON_ZERO_COMPLEXES)
    ):
        return COMPLEXES

    return UNKNOWN_VALUE_SET


def _infer_floor_divide_value_set(numerator_set: ValueSet, denominator_set: ValueSet) -> ValueSet:

    if numerator_set == ZERO:
        return ZERO

    if (
        is_known_subset(numerator_set, NON_NEGATIVE_REALS)
        and is_known_subset(denominator_set, POSITIVE_REALS)
    ):
        return NATURALS_0

    if (
        is_known_subset(numerator_set, NON_POSITIVE_REALS)
        and is_known_subset(denominator_set, NEGATIVE_REALS)
    ):
        return NATURALS_0

    if (
        is_known_subset(numerator_set, NEGATIVE_REALS)
        and is_known_subset(denominator_set, POSITIVE_REALS)
    ):
        return NEGATIVE_INTEGERS

    if (
        is_known_subset(numerator_set, POSITIVE_REALS)
        and is_known_subset(denominator_set, NEGATIVE_REALS)
    ):
        return NEGATIVE_INTEGERS

    if (
        is_known_subset(numerator_set, REALS)
        and is_known_subset(denominator_set, NON_ZERO_REALS)
    ):
        return INTEGERS

    return UNKNOWN_VALUE_SET


def _infer_modulo_value_set(dividend_set: ValueSet, divisor_set: ValueSet) -> ValueSet:

    if dividend_set == ZERO:
        return ZERO

    if (
        is_known_subset(dividend_set, INTEGERS)
        and is_known_subset(divisor_set, POSITIVE_INTEGERS)
    ):
        return NATURALS_0

    if (
        is_known_subset(dividend_set, INTEGERS)
        and is_known_subset(divisor_set, NEGATIVE_INTEGERS)
    ):
        return NON_POSITIVE_INTEGERS

    if (
        is_known_subset(dividend_set, REALS)
        and is_known_subset(divisor_set, POSITIVE_REALS)
    ):
        return NON_NEGATIVE_REALS

    if (
        is_known_subset(dividend_set, REALS)
        and is_known_subset(divisor_set, NEGATIVE_REALS)
    ):
        return NON_POSITIVE_REALS

    return UNKNOWN_VALUE_SET


def _infer_power_value_set(base_set: ValueSet, exponent_set: ValueSet) -> ValueSet:

    if exponent_set == ZERO:
        return ONE

    if exponent_set == ONE:
        return base_set

    if (
        is_known_subset(base_set, POSITIVE_REALS)
        and is_known_subset(exponent_set, REALS)
    ):
        return POSITIVE_REALS

    if (
        is_known_subset(base_set, NON_NEGATIVE_REALS)
        and is_known_subset(exponent_set, POSITIVE_REALS)
    ):
        return NON_NEGATIVE_REALS

    if (
        is_known_subset(base_set, REALS)
        and is_known_subset(exponent_set, POSITIVE_EVEN_INTEGERS)
    ):
        return NON_NEGATIVE_REALS

    if (
        is_known_subset(base_set, NON_ZERO_REALS)
        and is_known_subset(exponent_set, NEGATIVE_EVEN_INTEGERS)
    ):
        return POSITIVE_REALS

    if (
        is_known_subset(base_set, REALS)
        and is_known_subset(exponent_set, POSITIVE_ODD_INTEGERS)
    ):
        return REALS

    if (
        is_known_subset(base_set, NON_ZERO_REALS)
        and is_known_subset(exponent_set, NEGATIVE_ODD_INTEGERS)
    ):
        return NON_ZERO_REALS

    if (
        is_known_subset(base_set, NEGATIVE_REALS)
        and is_known_subset(exponent_set, NON_INTEGER_REALS)
    ):
        return COMPLEXES - REALS

    if is_known_subset(base_set, NON_ZERO_COMPLEXES):
        return NON_ZERO_COMPLEXES

    return COMPLEXES


# Unary operations

def _infer_negative_value_set(value_set: ValueSet) -> ValueSet:

    if value_set == ZERO:
        return ZERO

    if is_known_subset(value_set, POSITIVE_REALS):
        return NEGATIVE_REALS

    if is_known_subset(value_set, NEGATIVE_REALS):
        return POSITIVE_REALS

    if is_known_subset(value_set, NON_NEGATIVE_REALS):
        return NON_POSITIVE_REALS

    if is_known_subset(value_set, NON_POSITIVE_REALS):
        return NON_NEGATIVE_REALS

    if is_known_subset(value_set, INTEGERS):
        return INTEGERS

    if is_known_subset(value_set, REALS):
        return REALS

    if is_known_subset(value_set, COMPLEXES):
        return COMPLEXES

    return UNKNOWN_VALUE_SET


def _infer_absolute_value_set(value_set: ValueSet) -> ValueSet:

    if value_set == ZERO:
        return ZERO

    if is_known_subset(value_set, NON_NEGATIVE_REALS):
        return value_set

    if is_known_subset(value_set, NON_ZERO_REALS):
        return POSITIVE_REALS

    if is_known_subset(value_set, REALS):
        return NON_NEGATIVE_REALS

    if is_known_subset(value_set, NON_ZERO_COMPLEXES):
        return POSITIVE_REALS

    if is_known_subset(value_set, COMPLEXES):
        return NON_NEGATIVE_REALS

    return UNKNOWN_VALUE_SET