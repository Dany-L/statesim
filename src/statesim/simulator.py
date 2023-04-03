from typing import List, Tuple, Dict, Any, Optional, Union
from numpy.typing import NDArray
import numpy as np
import dataclasses

from scipy.integrate import solve_ivp
from .model.statespace import StateSpaceModel


@dataclasses.dataclass
class SimulationData:
    xs: List[NDArray[np.float64]]
    us: List[NDArray[np.float64]]
    ys: List[NDArray[np.float64]]
    t: NDArray[np.float64]
    name: str


class DiscreteSimulator:
    def __init__(self, T: float, step_size: float) -> None:
        assert T > step_size
        self.T = T
        self.step_size = step_size
        self.teval = np.linspace(
            0, self.T - step_size, int(self.T / step_size)
        )

    def simulate(
        self,
        model: StateSpaceModel,
        initial_state: NDArray[np.float64],
        input: List[NDArray[np.float64]],
        name: str = 'unknown',
        x_bar: Optional[NDArray[np.float64]] = None,
    ) -> SimulationData:
        x_bar = get_x_bar(x_bar=x_bar, nx=model._nx)
        xs = []
        xs.append(initial_state - x_bar)
        for k, _ in enumerate(self.teval):
            xs.append(model.state_dynamics(x=xs[k], u=input[k]))
        xs = [x + x_bar for x in xs]
        ys = model.output_layer(xs=xs, us=input)

        return SimulationData(
            xs=xs[0:-1], us=input, ys=ys, name=name, t=self.teval
        )


class ContinuousSimulator:
    def __init__(
        self, T: float, method: str = "RK45", step_size: float = 1.0
    ) -> None:
        assert T > step_size
        self.T = T
        self.method = method
        self.step_size = step_size
        self.teval = np.linspace(
            0, self.T - step_size, int(self.T / step_size)
        )

    def simulate(
        self,
        model: StateSpaceModel,
        initial_state: NDArray[np.float64],
        input: List[NDArray[np.float64]],
        name: str = 'unknown',
        x_bar: Optional[NDArray[np.float64]] = None,
    ) -> Tuple[SimulationData, Dict[str, Any]]:
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
        ys = model.output_layer(xs=xs, us=input)

        return (
            SimulationData(
                xs=xs, us=input, ys=ys, name=name, t=np.array(sol.t)
            ),
            sol,
        )


def get_x_bar(
    x_bar: Union[None, NDArray[np.float64]], nx: int
) -> NDArray[np.float64]:
    if x_bar is not None:
        x_bar = x_bar
    else:
        x_bar = np.zeros(shape=(nx, 1))
    return x_bar
