from __future__ import annotations

import sympy as sp

from src.problab.operations import AND, INVERT, OR
from src.problab.random_variables.nodes import Node, OperationNode


class Event:
    def __init__(self, node: Node[bool]):
        self._node: Node[bool] = node

    def __and__(self, other: Event) -> Event:

        if not type(other) is Event:
            return NotImplemented

        return Event(OperationNode(
            operation=AND,
            inputs=(self._node, other._node),
            value_set=sp.FiniteSet(False, True),
        ))

    def __or__(self, other: Event) -> Event:

        if not type(other) is Event:
            return NotImplemented

        return Event(OperationNode(
            operation=OR,
            inputs=(self._node, other._node),
            value_set=sp.FiniteSet(False, True),
        ))

    def __invert__(self) -> Event:

        return Event(OperationNode(
            operation=INVERT,
            inputs=(self._node,),
            value_set=sp.FiniteSet(False, True),
        ))
