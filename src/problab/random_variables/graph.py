from typing import Any
import networkx as nx

from src.problab.random_variables.nodes import Node


class NodeGraph:

    def __init__(self,
                 root_node: Node[Any],
                 max_size: int
                 ) -> None:

        if max_size < 0:
            raise ValueError("max_size must be non-negative")

        self._root_node = root_node
        self._max_size = max_size
        self._graph = nx.MultiDiGraph()
        self._is_complete = True
        self._build_graph()


    @property
    def size(self) -> int:
        # number of nodes in graph
        return self._graph.number_of_nodes()

    @property
    def is_complete(self) -> bool:
        return self._is_complete

    def _build_graph(self) -> None:
        self._add_node_recursive(self._root_node)

    def _add_node_recursive(self, node: Node[Any]) -> None:

        if node in self._graph:
            return

        if self._graph.number_of_nodes() >= self._max_size:
            self._is_complete = False
            return

        self._graph.add_node(node)

        for child_node in node.dependencies:
            self._add_node_recursive(child_node)

            if child_node in self._graph:
                self._graph.add_edge(node, child_node)


