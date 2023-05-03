from typing import List, Optional, Union
from numpy.typing import NDArray
import numpy as np
import dataclasses

from scipy.integrate import solve_ivp
from .model.statespace import StateSpaceModel
from .noise import get_noise, NoiseGeneration
import abc


@dataclasses.dataclass
class SimulationData:
    xs: List[NDArray[np.float64]]
    us: List[NDArray[np.float64]]
    ys: List[NDArray[np.float64]]
    t: NDArray[np.float64]
    name: str


class BasicSimulator(metaclass=abc.ABCMeta):
    def __init__(
        self,
        T: float,
        step_size: float,
    ) -> None:
        self.T = T
        self.step_size = step_size
        self.teval = np.linspace(
            0, self.T - step_size, int(self.T / step_size)
        )

    @abc.abstractmethod
    def simulate(
        self,
        model: StateSpaceModel,
        initial_state: NDArray[np.float64],
        input: List[NDArray[np.float64]],
        name: str = 'unknown',
        x_bar: Optional[NDArray[np.float64]] = None,
        noise_config: Optional[NoiseGeneration] = None,
    ) -> SimulationData:
        pass

    def output_layer(
        self,
        model: StateSpaceModel,
        xs: List[NDArray[np.float64]],
        us: List[NDArray[np.float64]],
        noise_config: Optional[NoiseGeneration] = None,
    ) -> List[NDArray[np.float64]]:
        ys = model.output_layer(xs=xs, us=us)
        if noise_config:
            noises = get_noise(
                size=model._ny, lenght=len(self.teval), config=noise_config
            )
            return [y + noise for y, noise in zip(ys, noises)]
        else:
            return ys


class DiscreteSimulator(BasicSimulator):
    def __init__(
        self,
        T: float,
        step_size: float,
    ) -> None:
        assert T > step_size
        super().__init__(T=T, step_size=step_size)
        self.step_size = step_size

    def simulate(
        self,
        model: StateSpaceModel,
        initial_state: NDArray[np.float64],
        input: List[NDArray[np.float64]],
        name: str = 'unknown',
        x_bar: Optional[NDArray[np.float64]] = None,
        noise_config: Optional[NoiseGeneration] = None,
    ) -> SimulationData:
        x_bar = get_x_bar(x_bar=x_bar, nx=model._nx)
        xs = []
        xs.append(initial_state - x_bar)
        for k, _ in enumerate(self.teval):
            xs.append(model.state_dynamics(x=xs[k], u=input[k]))
        xs = [x + x_bar for x in xs]

        ys = super().output_layer(
            model=model, xs=xs, us=input, noise_config=noise_config
        )

        return SimulationData(
            xs=xs[0:-1], us=input, ys=ys, name=name, t=self.teval
        )


class ContinuousSimulator(BasicSimulator):
    def __init__(
        self, T: float, step_size: float = 1.0, method: str = "RK45"
    ) -> None:
        assert T > step_size
        super().__init__(T=T, step_size=step_size)
        self.method = method

    def simulate(
        self,
        model: StateSpaceModel,
        initial_state: NDArray[np.float64],
        input: List[NDArray[np.float64]],
        name: str = 'unknown',
        x_bar: Optional[NDArray[np.float64]] = None,
        noise_config: Optional[NoiseGeneration] = None,
    ) -> SimulationData:
        x_bar = get_x_bar(x_bar=x_bar, nx=model._nx)
        sol = solve_ivp(
            fun=lambda t, y: np.squeeze(
                model.state_dynamics(u=input[int(t / self.step_size)], x=y)
            ),
            t_span=[0, self.T - self.step_size],
            y0=np.squeeze(initial_state - x_bar),
            method=self.method,
            t_eval=self.teval,
        )
        xs = [x.reshape(model._nx, 1) + x_bar for x in sol.y.T]

        ys = super().output_layer(
            model=model, xs=xs, us=input, noise_config=noise_config
        )

        return SimulationData(
            xs=xs, us=input, ys=ys, name=name, t=np.array(sol.t)
        )


def get_x_bar(
    x_bar: Union[None, NDArray[np.float64]], nx: int
) -> NDArray[np.float64]:
    if x_bar is not None:
        x_bar = x_bar
    else:
        x_bar = np.zeros(shape=(nx, 1))
    return x_bar
