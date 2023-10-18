import numpy as np
from .base import DynamicSystem
from numpy.typing import NDArray
from typing import Tuple, List
from sympy import symbols, diff, Matrix, lambdify, Piecewise
import sympy as sym


class CoupledMsd(DynamicSystem):
    """Coupled mass spring damper system"""

    def __init__(
        self,
        N: int = 4,
        k: NDArray[np.float64] = np.array([1.0, 5 / 6, 2 / 3, 1 / 2]),
        c: NDArray[np.float64] = np.array([1 / 4, 1 / 3, 5 / 12, 1 / 2]),
        m: NDArray[np.float64] = np.array([1 / 4, 1 / 3, 5 / 12, 1 / 2]),
    ) -> None:
        super().__init__()
        assert N >= 2
        self.nx = N * 2
        self.nu = 1
        self.ny = 1
        self.N = N
        self.k = k
        self.c = c
        self.m = m

        self._sym_dict = {}
        subs_dict = {}
        self._state_list = []
        for cart_idx in range(N):
            # states
            self._sym_dict[f'd{cart_idx}'] = symbols(
                f'd{cart_idx}'
            )  # displacement
            self._sym_dict[f'v{cart_idx}'] = symbols(
                f'v{cart_idx}'
            )  # velocity
            # parameters
            self._sym_dict[f'k{cart_idx}'] = symbols(f'k{cart_idx}')
            self._sym_dict[f'c{cart_idx}'] = symbols(f'c{cart_idx}')
            self._sym_dict[f'm{cart_idx}'] = symbols(f'm{cart_idx}')

            # create dictionary for parameter evaluation
            subs_dict[self._sym_dict[f'k{cart_idx}']] = self.k[cart_idx]
            subs_dict[self._sym_dict[f'c{cart_idx}']] = self.c[cart_idx]
            subs_dict[self._sym_dict[f'm{cart_idx}']] = self.m[cart_idx]

            # create state list
            self._state_list.append(self._sym_dict[f'd{cart_idx}'])
            self._state_list.append(self._sym_dict[f'v{cart_idx}'])

        self._sym_dict['u'] = symbols('u')
        self._state_list.append(self._sym_dict['u'])
        x_dot = self._get_nonlinear_function()
        self._f_symbol = Matrix([[x_dot_i] for x_dot_i in x_dot])
        self._f = lambdify(
            self._state_list, self._f_symbol.evalf(subs=subs_dict), 'numpy'
        )

    def _get_nonlinear_function(self) -> List[sym.core.add.Add]:
        def Gamma(d):
            return Piecewise(
                (d + 0.75, d <= -1), (0.25 * d, d < 1), (d - 0.75, True)
            )

        x_dot = []
        # first cart
        x_dot.append(self._sym_dict['v0'])
        x_dot.append(
            1
            / self._sym_dict['m0']
            * (
                self._sym_dict['u']
                + self._sym_dict['k0'] * Gamma(-self._sym_dict['d0'])
                + self._sym_dict['k1']
                * Gamma(self._sym_dict['d1'] - self._sym_dict['d0'])
                + self._sym_dict['c0'] * (-self._sym_dict['v0'])
                + self._sym_dict['c1']
                * (self._sym_dict['v1'] - self._sym_dict['v0'])
            )
        )
        # middle cart
        if self.N > 2:
            for state_idx in range(1, self.N - 1):
                x_dot.append(self._sym_dict[f'v{state_idx}'])
                x_dot.append(
                    1
                    / self._sym_dict[f'm{state_idx}']
                    * (
                        +self._sym_dict[f'k{state_idx}']
                        * Gamma(
                            self._sym_dict[f'd{state_idx-1}']
                            - self._sym_dict[f'd{state_idx}']
                        )
                        + self._sym_dict[f'k{state_idx+1}']
                        * Gamma(
                            self._sym_dict[f'd{state_idx+1}']
                            - self._sym_dict[f'd{state_idx}']
                        )
                        + self._sym_dict[f'c{state_idx}']
                        * (
                            self._sym_dict[f'v{state_idx-1}']
                            - self._sym_dict[f'v{state_idx}']
                        )
                        + self._sym_dict[f'c{state_idx+1}']
                        * (
                            self._sym_dict[f'v{state_idx+1}']
                            - self._sym_dict[f'v{state_idx}']
                        )
                    )
                )
        # last cart
        state_idx = self.N - 1
        x_dot.append(self._sym_dict[f'v{state_idx}'])
        x_dot.append(
            1
            / self._sym_dict[f'm{state_idx}']
            * (
                +self._sym_dict[f'k{state_idx}']
                * Gamma(
                    self._sym_dict[f'd{state_idx-1}']
                    - self._sym_dict[f'd{state_idx}']
                )
                + self._sym_dict[f'c{state_idx}']
                * (
                    self._sym_dict[f'v{state_idx-1}']
                    - self._sym_dict[f'v{state_idx}']
                )
            )
        )

        return x_dot

    def state_dynamics(self, x, u):
        states = (x_i for x_i in x)
        return self._f(*states, np.squeeze(u)).reshape(self.nx, 1)

    def output_function(self, x, u):
        C = np.zeros(shape=(1, self.nx))
        C[0, -2] = 1
        return C @ x

    def get_linearization(
        self,
    ) -> Tuple[
        sym.matrices.dense.MutableDenseMatrix,
        sym.matrices.dense.MutableDenseMatrix,
    ]:
        A = []
        B = []
        for row in range(2 * self.N):
            column_list = []
            for state_idx in range(self.N):
                column_list.append(
                    diff(
                        self._f_symbol[row, 0], self._sym_dict[f'd{state_idx}']
                    )
                )
                column_list.append(
                    diff(
                        self._f_symbol[row, 0], self._sym_dict[f'v{state_idx}']
                    )
                )
            A.append(column_list)
            B.append([diff(self._f_symbol[row, 0], self._sym_dict['u'])])

        return Matrix(A), Matrix(B)

    def evaluate_linearization(
        self,
        A_sym: sym.matrices.dense.MutableDenseMatrix,
        B_sym: sym.matrices.dense.MutableDenseMatrix,
        x_bar: NDArray[np.float64],
        u_bar: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        eval_dict = {}
        for cart_idx in range(self.N):
            eval_dict[self._sym_dict[f'd{cart_idx}']] = x_bar[cart_idx * 2, 0]
            eval_dict[self._sym_dict[f'v{cart_idx}']] = x_bar[
                cart_idx * 2 + 1, 0
            ]

            eval_dict[self._sym_dict[f'k{cart_idx}']] = self.k[cart_idx]
            eval_dict[self._sym_dict[f'c{cart_idx}']] = self.c[cart_idx]
            eval_dict[self._sym_dict[f'm{cart_idx}']] = self.m[cart_idx]

        eval_dict[self._sym_dict['u']] = u_bar
        A = A_sym.evalf(subs=eval_dict)
        B = B_sym.evalf(subs=eval_dict)
        return np.array(A, dtype=np.float64), np.array(B, dtype=np.float64)
