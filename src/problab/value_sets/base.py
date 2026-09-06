from typing import TypeAlias
import sympy as sp

n = sp.Symbol("n", integer=True)

class _UnknownValueSet:

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNKNOWN_VALUE_SET"

ValueSet: TypeAlias = sp.Set | _UnknownValueSet

