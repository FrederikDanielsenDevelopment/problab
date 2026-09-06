import sympy as sp

from src.problab.value_sets.base import ValueSet
from src.problab.value_sets.sets import UNKNOWN_VALUE_SET

def is_known_subset(subset: ValueSet, superset: sp.Set) -> bool:
    if subset is UNKNOWN_VALUE_SET:
        return False

    return subset.is_subset(superset) is True


