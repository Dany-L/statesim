import abc
from numpy.typing import NDArray
from typing import List, Callable
import numpy as np


class StateSpaceModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self) -> None:
        self._nx: int = 0
        self._ny: int = 0
        self._nu: int = 0

    @abc.abstractmethod
    def state_dynamics(
        self,
        x: NDArray[np.float64],
        u: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        pass

    @abc.abstractmethod
    def output_layer(
        self, xs: List[NDArray[np.float64]], us: List[NDArray[np.float64]]
    ) -> List[NDArray[np.float64]]:
        pass


class Linear(StateSpaceModel):
    def __init__(
        self,
        A: NDArray[np.float64],
        B: NDArray[np.float64],
        C: NDArray[np.float64],
        D: NDArray[np.float64],
    ) -> None:
        self._nx = A.shape[0]
        self._nu = B.shape[1]
        self._ny = C.shape[0]

        self.A = A  # state matrix
        self.B = B  # input matrix
        self.C = C  # output matrix
        self.D = D  # direct feed through term

    def state_dynamics(self, x, u):
        x = x.reshape(self._nx, 1)
        u = u.reshape(self._nu, 1)
        return self.A @ x + self.B @ u

    def output_layer(self, xs, us):
        return [self.C @ x + self.D @ u for x, u in zip(xs, us)]


class Nonlinear(StateSpaceModel):
    def __init__(
        self,
        f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        g: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        nx: int,
        ny: int,
        nu: int,
    ) -> None:
        self.f = f  # nonlinear state function
        self.g = g  # nonlinear output function
        self._nx = nx
        self._ny = ny
        self._nu = nu

    def state_dynamics(self, x, u):
        return self.f(x, u)

    def output_layer(self, xs, us):
        return [self.g(x, u) for x, u in zip(xs, us)]


class Lure(Linear):
    def __init__(
        self,
        A: NDArray[np.float64],
        B1: NDArray[np.float64],
        B2: NDArray[np.float64],
        C1: NDArray[np.float64],
        C2: NDArray[np.float64],
        D11: NDArray[np.float64],
        D12: NDArray[np.float64],
        D21: NDArray[np.float64],
        Delta: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ) -> None:
        super().__init__(A=A, B=B1, C=C1, D=D11)
        self.B2 = B2
        self.C2 = C2
        self.D12 = D12
        self.D21 = D21
        self.Delta = Delta  # static nonlinearity

    def state_dynamics(self, x, u):
        return (
            super().state_dynamics(x, u)
            + self.B2 @ self.Delta(self.C2 @ x + self.D21 @ u)
        ).reshape(self._nx, 1)

    def output_layer(self, xs, us):
        internals = super().output_layer(xs, us)

        return [
            (
                internal + self.D12 @ self.Delta(self.C2 @ x + self.D21 @ u)
            ).reshape(self._ny, 1)
            for internal, u, x in zip(internals, us, xs)
        ]
