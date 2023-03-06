from typing import List, Tuple, Dict, Any
from numpy.typing import NDArray
import numpy as np
import dataclasses

from scipy.integrate import solve_ivp

# import model
from .model.statespace import StateSpaceModel


@dataclasses.dataclass
class SimulationResult:
    xs: List[NDArray[np.float64]]
    us: List[NDArray[np.float64]]
    ys: List[NDArray[np.float64]]
    name: str


class Simulator:
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
    ) -> Tuple[SimulationResult, NDArray[np.float64], Dict[str, Any],]:
        sol = solve_ivp(
            fun=lambda t, y: np.squeeze(
                model.state_dynamics(u=input[int(t / self.step_size)], x=y)
            ),
            t_span=[0, self.T - self.step_size],
            y0=np.squeeze(initial_state),
            method=self.method,
            t_eval=self.teval,
        )
        xs = [x.reshape(model._nx, 1) for x in sol.y.T]
        ys = model.output_layer(xs=xs, us=input)

        return (
            SimulationResult(xs=xs, us=input, ys=ys, name=name),
            np.array(sol.t),
            sol,
        )
