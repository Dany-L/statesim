from statesim.model.statespace import (
    Linear,
    Nonlinear,
)
from statesim.simulator import ContinuousSimulator, DiscreteSimulator
from statesim.system.cartpole import CartPole
import utils
import numpy as np
from statesim.io import read_measurement_csv
import os

DIRNAME = os.path.dirname(__file__)


class TestSimulation:
    """Compare simulation results to matlab generated simulation

    Load data from .csv files and run comparison
    """

    T = 10.0
    step_size = 0.01
    x_bar = utils.get_linearization_point_cartpole()
    x0 = np.array([[0], [0], [np.pi + 0.1], [0]])

    def test_cartpole_compare_simulation_results_continuous_linear(
        self,
    ) -> None:
        filepath = os.path.join(
            DIRNAME, 'data/2023_03_09-07_56_08_cartpole_linear_continous.csv'
        )
        measurements = read_measurement_csv(filepath=filepath)

        sys = CartPole(mu_p=0.01)
        A_sym, B_sym = sys.get_linearization()
        A, B = sys.evaluate_linearization(
            A_sym=A_sym, B_sym=B_sym, x_bar=self.x_bar, u_bar=np.array([[0]])
        )
        model = Linear(A=A, B=B, C=np.array([[0, 0, 1, 0]]), D=np.array([[0]]))
        sim = ContinuousSimulator(T=self.T, step_size=self.step_size)
        result, _ = sim.simulate(
            model=model,
            initial_state=self.x0,
            input=measurements.us,
            name='linear continuous',
            x_bar=self.x_bar,
        )
        error = utils.calculate_error(result.ys, measurements.ys)
        assert error < 1e-4

    def test_cartpole_compare_simulation_results_continuous_nonlinear(
        self,
    ) -> None:
        filepath = os.path.join(
            DIRNAME,
            'data/2023_03_09-07_56_08_cartpole_nonlinear_continuous.csv',
        )
        measurements = read_measurement_csv(filepath=filepath)

        sys = CartPole(mu_p=0.01)
        model = Nonlinear(
            f=sys.state_dynamics,
            g=sys.output_function,
            nx=sys.nx,
            nu=sys.nu,
            ny=sys.ny,
        )
        sim = ContinuousSimulator(T=self.T, step_size=self.step_size)
        result, _ = sim.simulate(
            model=model,
            initial_state=self.x0,
            input=measurements.us,
            name='nonlinear continuous',
        )
        error = utils.calculate_error(result.ys, measurements.ys)
        assert error < 1e-4

    def test_cartpole_compare_simulation_results_discrete_linear(self) -> None:
        filepath = os.path.join(
            DIRNAME, 'data/2023_03_09-07_56_08_cartpole_linear_discrete.csv'
        )
        measurements = read_measurement_csv(filepath=filepath)

        sys = CartPole(mu_p=0.01)
        A_sym, B_sym = sys.get_linearization()
        A, B = sys.evaluate_linearization(
            A_sym=A_sym, B_sym=B_sym, x_bar=self.x_bar, u_bar=np.array([[0]])
        )
        model = Linear(
            A=A * self.step_size + np.eye(sys.nx),
            B=B * self.step_size,
            C=np.array([[0, 0, 1, 0]]),
            D=np.array([[0]]),
        )
        sim = DiscreteSimulator(T=self.T, step_size=self.step_size)
        result = sim.simulate(
            model=model,
            initial_state=self.x0,
            input=measurements.us,
            name='linear discrete',
            x_bar=self.x_bar,
        )
        error = utils.calculate_error(result.ys, measurements.ys)

        assert error < 1e-4
