import numpy as np
import abc
from numpy.typing import NDArray
from typing import Tuple
import sympy as sym


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

    @abc.abstractmethod
    def get_linearization(
        self,
    ) -> Tuple[
        sym.matrices.dense.MutableDenseMatrix,
        sym.matrices.dense.MutableDenseMatrix,
    ]:
        pass

    @abc.abstractmethod
    def evaluate_linearization(
        self,
        A_sym: sym.matrices.dense.MutableDenseMatrix,
        B_sym: sym.matrices.dense.MutableDenseMatrix,
        x_bar: NDArray[np.float64],
        u_bar: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        pass
