from typing import Any, TypeVar

import numpy as np

from src.problab.random_variables.nodes import Node

T = TypeVar('T')


class RealizationContext:

    def __init__(
        self,
        num_samples: int = 1,
        rng: np.random.Generator | None = None,
    ) -> None:

        if num_samples < 1:
            raise ValueError("'num_samples' must be at least 1.")

        self._num_samples = num_samples
        self._rng = np.random.default_rng() if rng is None else rng
        self._realizations: dict[Node[Any], np.ndarray] = {}

    def __contains__(self, item):
        return item in self._realizations.keys()

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    def evaluate(self, node: Node[T]) -> np.ndarray:
        if node not in self._realizations:
            self._realizations[node] = node._evaluate(self)

        return self._realizations[node]


