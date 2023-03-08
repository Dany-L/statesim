from hybridstablemodel.model.statespace import (
    Linear,
    Nonlinear,
)
from hybridstablemodel.simulator import ContinuousSimulator, DiscreteSimulator
from hybridstablemodel.plot.plot_simulation_results import plot_comparison
from typing import List, Dict
import utils
import numpy as np


class TestClass:
    def test_linear_continuous_simulator(self) -> None:
        u = utils.get_input()
        x0 = utils.get_initial_state()
        A, B, C, D = utils.get_linear_matrices()
        nx = utils.get_state_size()
        nu = utils.get_input_size()
        ny = utils.get_output_size()
        model = Linear(A=A, B=B, C=C, D=D)

        sim = ContinuousSimulator(T=float(len(u)))
        result, info = sim.simulate(model=model, initial_state=x0, input=u)

        assert isinstance(result.ys, List)
        assert isinstance(info, Dict)
        assert isinstance(result.teval, np.ndarray)
        assert isinstance(result.ys[0], np.ndarray)
        # output
        assert len(result.ys) == len(u)
        assert result.ys[0].shape == (ny, 1)
        # state
        assert len(result.xs) == len(u)
        assert result.xs[0].shape == (nx, 1)
        # input
        assert len(result.us) == len(u)
        assert result.us[0].shape == (nu, 1)

    def test_linear_continuous_simulator_step_size(self) -> None:
        u = utils.get_input()
        x0 = utils.get_initial_state()
        A, B, C, D = utils.get_linear_matrices()
        step_size = 0.02
        model = Linear(A=A, B=B, C=C, D=D)
        sim = ContinuousSimulator(
            T=float(len(u) * step_size), step_size=step_size
        )
        result, info = sim.simulate(model=model, initial_state=x0, input=u)

        assert info.success is True
        assert (result.teval[2] - result.teval[1]) - step_size < 1e-5

    def test_linear_model(self) -> None:
        u = utils.get_input()
        x0 = utils.get_initial_state()
        A, B, C, D = utils.get_linear_matrices()
        model = Linear(A=A, B=B, C=C, D=D)
        xdot = model.state_dynamics(x0, u[0])

        assert isinstance(xdot, np.ndarray)
        assert xdot.shape == (2, 1)

    def test_nonlinear_continuous_simulator(self) -> None:
        u = utils.get_input()
        x0 = utils.get_initial_state()
        nx = utils.get_state_size()
        nu = utils.get_input_size()
        ny = utils.get_output_size()
        model = Nonlinear(
            f=utils.get_nonlinear_state_function(),
            g=utils.get_nonlinear_output_function(),
            nu=nu,
            nx=nx,
            ny=ny,
        )
        sim = ContinuousSimulator(T=float(len(u)))
        result, info = sim.simulate(model=model, initial_state=x0, input=u)

        assert isinstance(result.ys, List)
        assert isinstance(info, Dict)
        assert isinstance(result.teval, np.ndarray)
        assert isinstance(result.ys[0], np.ndarray)
        # output
        assert len(result.ys) == len(u)
        assert result.ys[0].shape == (ny, 1)
        # state
        assert len(result.xs) == len(u)
        assert result.xs[0].shape == (nx, 1)
        # input
        assert len(result.us) == len(u)
        assert result.us[0].shape == (nu, 1)

    def test_nonlinear_continuous_simulator_step_size(self) -> None:
        u = utils.get_input()
        x0 = utils.get_initial_state()
        nx = utils.get_state_size()
        nu = utils.get_input_size()
        ny = utils.get_output_size()
        step_size = 0.02
        model = Nonlinear(
            f=utils.get_nonlinear_state_function(),
            g=utils.get_nonlinear_output_function(),
            nu=nu,
            nx=nx,
            ny=ny,
        )
        sim = ContinuousSimulator(
            T=float(len(u) * step_size), step_size=step_size
        )
        result, info = sim.simulate(model=model, initial_state=x0, input=u)

        assert info.success is True
        assert (result.teval[2] - result.teval[1]) - step_size < 1e-5

    def test_nonlinear_model(self) -> None:
        u = utils.get_input()
        x0 = utils.get_initial_state()
        nx = utils.get_state_size()
        nu = utils.get_input_size()
        ny = utils.get_output_size()
        model = Nonlinear(
            f=utils.get_nonlinear_state_function(),
            g=utils.get_nonlinear_output_function(),
            nu=nu,
            nx=nx,
            ny=ny,
        )
        xdot = model.state_dynamics(u=u[0], x=x0)

        assert isinstance(xdot, np.ndarray)
        assert xdot.shape == (2, 1)

    def test_linear_discrete_simulator(self) -> None:
        u = utils.get_input()
        x0 = utils.get_initial_state()
        A, B, C, D = utils.get_linear_matrices()
        nx = utils.get_state_size()
        nu = utils.get_input_size()
        ny = utils.get_output_size()
        model = Linear(A=A, B=B, C=C, D=D)
        step_size = 0.02

        sim = DiscreteSimulator(
            T=float(len(u) * step_size), step_size=step_size
        )
        result = sim.simulate(model=model, initial_state=x0, input=u)

        print(result.xs[0])
        assert isinstance(result.ys, List)
        assert isinstance(result.teval, np.ndarray)
        assert isinstance(result.ys[0], np.ndarray)
        # output
        assert len(result.ys) == len(u)
        assert result.ys[0].shape == (ny, 1)
        # state
        assert len(result.xs) == len(u)
        assert result.xs[0].shape == (nx, 1)
        # input
        assert len(result.us) == len(u)
        assert result.us[0].shape == (nu, 1)

    def test_plot_simulation_results(self) -> None:
        types = ['xs', 'us', 'ys']
        results = utils.get_simulation_results()
        for type in types:
            plot_comparison(results=results, type=type)