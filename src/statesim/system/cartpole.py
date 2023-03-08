import numpy as np
import abc
from numpy.typing import NDArray
from typing import Tuple, List
from sympy import symbols, diff, sin, cos, sign, Matrix, lambdify
import sympy as sym

# SympyType = TypeVar('SympyType', bound=sym.core)
# SympyMatrix = TypeVar('SympyMatrix', bound=sym.matrices)


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


class CartPole(DynamicSystem):
    def __init__(
        self,
        g: float = 9.81,
        m_c: float = 1.0,
        m_p: float = 0.1,
        length: float = 0.5,
        mu_c: float = 0.0,
        mu_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.g = g
        self.m_c = m_c
        self.m_p = m_p
        self.length = length  # actually half the pole's length
        self.mu_c = mu_c  # coefficient of friction of cart on track
        self.mu_p = mu_p  # coefficient of friction of pole on cart
        self.nx = 4
        self.nu = 1
        self.ny = 1

        (
            self._x1,
            self._x2,
            self._x3,
            self._x4,
            self._u,
            self._g,
            self._m_p,
            self._m_c,
            self._length,
            self._mu_c,
            self._mu_p,
        ) = symbols('x1 x2 x3 x4 u g m_p m_c l mu_c mu_p')
        x_dot = self._get_nonlinear_function()
        self._f_symbol = Matrix(
            [[x_dot[0]], [x_dot[1]], [x_dot[2]], [x_dot[3]]]
        )

    def _get_nonlinear_function(self) -> List[sym.core.add.Add]:
        x4_dot = (
            self._g * sin(self._x3)
            + cos(self._x3)
            * (
                (
                    -self._u
                    - self._m_p * self._length * self._x4**2 * sin(self._x3)
                    + self._mu_c * sign(self._x2)
                )
                / (self._m_c + self._m_p)
            )
            - (self._mu_p * self._x4) / (self._m_p * self._length)
        ) / (
            self._length
            * (
                4 / 3
                - (self._m_p * cos(self._x3) ** 2) / (self._m_c + self._m_p)
            )
        )
        x1_dot = self._x2
        x2_dot = (
            self._u
            + self._m_p * self._length * self._x4**2 * sin(self._x3)
            - self._m_p * self._length * x4_dot * cos(self._x3)
            - self._mu_c * sign(self._x2)
        ) / (self._m_c + self._m_p)
        x3_dot = self._x4
        return [x1_dot, x2_dot, x3_dot, x4_dot]

    def state_dynamics(self, x, u):
        x1, x2, x3, x4 = x
        eval_dict = {
            self._g: self.g,
            self._m_p: self.m_p,
            self._m_c: self.m_c,
            self._length: self.length,
            self._mu_c: self.mu_c,
            self._mu_p: self.mu_p,
        }
        f = lambdify(
            [self._x1, self._x2, self._x3, self._x4, self._u],
            self._f_symbol.evalf(subs=eval_dict),
            'numpy',
        )
        return f(x1, x2, x3, x4, np.squeeze(u)).reshape(self.nx, 1)

    def output_function(self, x, u):
        return np.array([[0, 0, 1, 0]]) @ x

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
            self._x3: x_bar[2, 0],
            self._x4: x_bar[3, 0],
            self._u: u_bar[0, 0],
            self._g: self.g,
            self._m_p: self.m_p,
            self._m_c: self.m_c,
            self._length: self.length,
            self._mu_c: self.mu_c,
            self._mu_p: self.mu_p,
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
                    diff(self._f_symbol[0, 0], self._x3),
                    diff(self._f_symbol[0, 0], self._x4),
                ],
                [
                    diff(self._f_symbol[1, 0], self._x1),
                    diff(self._f_symbol[1, 0], self._x2),
                    diff(self._f_symbol[1, 0], self._x3),
                    diff(self._f_symbol[1, 0], self._x4),
                ],
                [
                    diff(self._f_symbol[2, 0], self._x1),
                    diff(self._f_symbol[2, 0], self._x2),
                    diff(self._f_symbol[2, 0], self._x3),
                    diff(self._f_symbol[2, 0], self._x4),
                ],
                [
                    diff(self._f_symbol[3, 0], self._x1),
                    diff(self._f_symbol[3, 0], self._x2),
                    diff(self._f_symbol[3, 0], self._x3),
                    diff(self._f_symbol[3, 0], self._x4),
                ],
            ],
        )

        B = Matrix(
            [
                [diff(self._f_symbol[0, 0], self._u)],
                [diff(self._f_symbol[1, 0], self._u)],
                [diff(self._f_symbol[2, 0], self._u)],
                [diff(self._f_symbol[3, 0], self._u)],
            ]
        )

        return A, B
