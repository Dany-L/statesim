import numpy as np
import abc
from numpy.typing import NDArray


class DynamicSystem(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def state_dynamics(
        self, x: NDArray[np.float64], u: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        pass

    @abc.abstractmethod
    def output_function(
        self, x: NDArray[np.float64], u: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        pass
