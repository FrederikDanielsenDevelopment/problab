from abc import ABC, abstractmethod
import numpy as np
from numbers import Real
from numpy.random import normal

class Distribution(ABC):

    @abstractmethod
    def sample(self, size: int | tuple[int, ...] | None = None, *, rng: np.random.Generator | None = None) -> Real | np.ndarray:
        pass


class DegenerateDistribution(Distribution):

    def __init__(self, value: Real):
        self.value = value

    def sample(self, size: int | tuple[int, ...] | None = None, *, rng: np.random.Generator | None = None) -> Real | np.ndarray:
        return np.full(size, self.value)

