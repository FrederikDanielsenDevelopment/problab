from abc import ABC, abstractmethod
import numpy as np
from numbers import Real

class Distribution(ABC):

    @abstractmethod
    def sample(self, size: int | tuple[int, ...] | None = None, *, rng: np.random.Generator | None = None) -> Real | np.ndarray:




class TrivialDistribution(Distribution):
