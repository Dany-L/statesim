import numpy as np
from .base import DynamicSystem
from numpy.typing import NDArray
from typing import Tuple, List
from sympy import symbols, diff, sin, Matrix, lambdify
import sympy as sym

class Pendulum(DynamicSystem):
    """Pendulum with torque input"""

    def __init__(
        self,
        k: float = 1.0
    ) -> None:
        super().__init__()
        self.k = k
        self.nx = 2
        self.nu = 1
        self.ny = 1

        (
            self._x1,
            self._x2,
            self._u,
            self._k
        ) = symbols('x1 x2 u k')
        x_dot = self._get_nonlinear_function()
        self._f_symbol = Matrix([[x_dot[0]], [x_dot[1]]])
        self._f = lambdify(
            [self._x1, self._x2, self._u],
            self._f_symbol.evalf(
                subs={
                    self._k: self.k,
                }
            ),
            'numpy',
        )

    def _get_nonlinear_function(self) -> List[sym.core.add.Add]:
        x1_dot = self._x2
        x2_dot = (-self._k * sin(self._x1) -1/100*self._x2 + self._u)
        return [x1_dot, x2_dot]

    def state_dynamics(self, x, u):
        x1, x2 = x
        return self._f(x1, x2, np.squeeze(u)).reshape(self.nx, 1)

    def output_function(self, x, u):
        return np.array([[1, 0]]) @ x

    def evaluate_linearization(
        self,
        A_sym: sym.matrices.dense.MutableDenseMatrix,
        B_sym: sym.matrices.dense.MutableDenseMatrix,
        x_bar: NDArray[np.float64],
        u_bar: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        eval_dict = {
            self._x1: x_bar[0, 0],
            self._x2: x_bar[1, 0],
            self._u: u_bar[0, 0],
            self._k: self.k
        }
        A = A_sym.evalf(subs=eval_dict)
        B = B_sym.evalf(subs=eval_dict)
        return np.array(A, dtype=np.float64), np.array(B, dtype=np.float64)

    def get_linearization(
        self,
    ) -> Tuple[
        sym.matrices.dense.MutableDenseMatrix,
        sym.matrices.dense.MutableDenseMatrix,
    ]:
        A = Matrix(
            [
                [
                    diff(self._f_symbol[0, 0], self._x1),
                    diff(self._f_symbol[0, 0], self._x2),
                ],
                [
                    diff(self._f_symbol[1, 0], self._x1),
                    diff(self._f_symbol[1, 0], self._x2),
                ],
            ],
        )

        B = Matrix(
            [
                [diff(self._f_symbol[0, 0], self._u)],
                [diff(self._f_symbol[1, 0], self._u)],
            ]
        )

        return A, B