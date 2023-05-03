from statesim.model.statespace import Linear, Nonlinear, Lure
from statesim.simulator import ContinuousSimulator, DiscreteSimulator
from statesim.analysis.plot_simulation_results import (
    plot_comparison,
    plot_inputs,
    plot_outputs,
    plot_states,
)
from statesim.noise import get_noise, NoiseGeneration
from statesim.analysis.system_analysis import SystemAnalysisContinuous
from statesim.system.cartpole import CartPole
from statesim.system.inverted_pendulum import InvertedPendulum
from statesim.system.coupled_msd import CoupledMsd
from statesim.io import (
    read_measurement_csv,
    write_measurement_csv,
)
from statesim.generate.input import generate_random_static_input
from typing import List, Dict
import utils
import numpy as np
import sympy as sym
import os
import math


def test_linear_continuous_simulator() -> None:
    u = utils.get_input()
    x0 = utils.get_initial_state()
    A, B, C, D = utils.get_stable_linear_matrices()
    nx = utils.get_state_size()
    nu = utils.get_input_size()
    ny = utils.get_output_size()
    model = Linear(A=A, B=B, C=C, D=D)

    sim = ContinuousSimulator(T=float(len(u)))
    measurement_noises = (NoiseGeneration('gaussian', 0.0, 0.01), None)
    for measurement_noise in measurement_noises:
        result = sim.simulate(
            model=model,
            initial_state=x0,
            input=u,
            noise_config=measurement_noise,
        )

        assert isinstance(result.ys, List)
        assert isinstance(result.t, np.ndarray)
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


def test_linear_continuous_simulator_step_size() -> None:
    u = utils.get_input()
    x0 = utils.get_initial_state()
    A, B, C, D = utils.get_stable_linear_matrices()
    step_size = 0.02
    model = Linear(A=A, B=B, C=C, D=D)
    sim = ContinuousSimulator(T=float(len(u) * step_size), step_size=step_size)
    result = sim.simulate(model=model, initial_state=x0, input=u)

    assert (result.t[2] - result.t[1]) - step_size < 1e-5


def test_linear_model() -> None:
    u = utils.get_input()
    x0 = utils.get_initial_state()
    A, B, C, D = utils.get_stable_linear_matrices()
    model = Linear(A=A, B=B, C=C, D=D)
    xdot = model.state_dynamics(x0, u[0])

    assert isinstance(xdot, np.ndarray)
    assert xdot.shape == (2, 1)


def test_nonlinear_continuous_simulator() -> None:
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
    result = sim.simulate(model=model, initial_state=x0, input=u)

    assert isinstance(result.ys, List)
    assert isinstance(result.t, np.ndarray)
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


def test_nonlinear_continuous_simulator_step_size() -> None:
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
    sim = ContinuousSimulator(T=float(len(u) * step_size), step_size=step_size)
    result = sim.simulate(model=model, initial_state=x0, input=u)

    assert (result.t[2] - result.t[1]) - step_size < 1e-5


def test_nonlinear_model() -> None:
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


def test_linear_discrete_simulator() -> None:
    u = utils.get_input()
    x0 = utils.get_initial_state()
    A, B, C, D = utils.get_stable_linear_matrices()
    nx = utils.get_state_size()
    nu = utils.get_input_size()
    ny = utils.get_output_size()
    model = Linear(A=A, B=B, C=C, D=D)
    step_size = 0.02

    sim = DiscreteSimulator(T=float(len(u) * step_size), step_size=step_size)
    measurement_noises = (NoiseGeneration('gaussian', 0.0, 0.01), None)
    for measurement_noise in measurement_noises:
        result = sim.simulate(
            model=model,
            initial_state=x0,
            input=u,
            noise_config=measurement_noise,
        )

        assert isinstance(result.ys, List)
        assert isinstance(result.t, np.ndarray)
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


def test_plot_simulation_results() -> None:
    types = ['xs', 'us', 'ys']
    results = utils.get_simulation_results()
    for type in types:
        plot_comparison(results=results, type=type)


def test_plot_states() -> None:
    results = utils.get_simulation_results()
    plot_states(result=results[0])


def test_plot_inputs() -> None:
    results = utils.get_simulation_results()
    plot_inputs(result=results[0])


def test_plot_outputs() -> None:
    results = utils.get_simulation_results()
    plot_outputs(result=results[0])


def test_cartpole_state_dynamics() -> None:
    system = CartPole()
    x0 = utils.get_initial_state_cartpole()
    x1 = system.state_dynamics(x=x0, u=np.array([[0]]))
    assert x1.shape == (4, 1)


def test_cartpole_linearization() -> None:
    system = CartPole()
    A, B = system.get_linearization()

    assert isinstance(A, sym.matrices.dense.MutableDenseMatrix)
    assert isinstance(B, sym.matrices.dense.MutableDenseMatrix)


def test_cartpole_linearization_evaluation() -> None:
    system = CartPole()
    A_sym, B_sym = system.get_linearization()
    x_bar = utils.get_linearization_point_cartpole()
    A, B = system.evaluate_linearization(
        A_sym=A_sym, B_sym=B_sym, x_bar=x_bar, u_bar=np.array([[0]])
    )

    assert isinstance(A, np.ndarray)
    assert A.shape == (4, 4)
    assert isinstance(B, np.ndarray)
    assert B.shape == (4, 1)


def test_read_measurement_csv() -> None:
    filepath = os.path.join(
        utils.get_directory(),
        'data/2023_03_09-01_25_33_cartpole_linear_continous.csv',
    )
    measurement = read_measurement_csv(filepath=filepath)

    # y
    assert measurement.ys[0].shape == (2, 1)
    # u
    assert measurement.us[0].shape == (1, 1)


def test_generate_static_random_input() -> None:
    us = generate_random_static_input(
        N=20, nu=2, amplitude_range=(-4.0, 4.0), frequency_range=(1, 4)
    )
    us_numpy = np.array(us)
    assert us[0].shape == (2, 1)
    assert len(us) == 20
    # check if elements are non zero
    assert np.squeeze(np.where(us_numpy == 0)).size == 0


def test_write_measurement_csv() -> None:
    data = utils.get_measurement_data()
    filepath = os.path.join(utils.get_tmp_directory(), 'measurement.csv')
    write_measurement_csv(filepath=filepath, simulation_data=data)


def test_plot_magnitude() -> None:
    ana = SystemAnalysisContinuous(utils.get_stable_linear_matrices())
    ana.plot_magnitude()


def test_get_peak_gain() -> None:
    ana = SystemAnalysisContinuous(utils.get_stable_linear_matrices())
    ana.get_peak_gain()


def test_get_h_inf_norm() -> None:
    ana = SystemAnalysisContinuous(utils.get_unstable_linear_matrices())

    h_inf = ana.get_h_inf_norm()
    print(h_inf)
    assert math.isinf(h_inf)

    ana = SystemAnalysisContinuous(utils.get_stable_linear_matrices())
    h_inf = ana.get_h_inf_norm()
    assert not math.isinf(h_inf)


def test_get_real_eigenvalues() -> None:
    ana = SystemAnalysisContinuous(utils.get_stable_linear_matrices())
    ana.get_real_eigenvalues()


def test_is_stable() -> None:
    ana = SystemAnalysisContinuous(utils.get_stable_linear_matrices())
    assert ana.is_stable()
    ana = SystemAnalysisContinuous(utils.get_unstable_linear_matrices())
    assert not ana.is_stable()


def test_analysis() -> None:
    ana = SystemAnalysisContinuous(utils.get_stable_linear_matrices())
    ana.analysis()
    ana = SystemAnalysisContinuous(utils.get_unstable_linear_matrices())
    ana.analysis()


def test_inverted_pendulum_state_dynamics() -> None:
    system = InvertedPendulum()
    x0 = utils.get_initial_state_inverted_pendulum()
    x1 = system.state_dynamics(x=x0, u=np.array([[0]]))
    assert x1.shape == (2, 1)


def test_inverted_pendulum_linearization_evaluation() -> None:
    system = InvertedPendulum()
    A_sym, B_sym = system.get_linearization()
    x_bar = utils.get_linearization_point_inverted_pendulum()
    A, B = system.evaluate_linearization(
        A_sym=A_sym, B_sym=B_sym, x_bar=x_bar, u_bar=np.array([[0]])
    )

    assert isinstance(A, np.ndarray)
    assert A.shape == (2, 2)
    assert isinstance(B, np.ndarray)
    assert B.shape == (2, 1)


def test_coupled_msd_dynamics() -> None:
    system = CoupledMsd()
    x0 = utils.get_initial_state_msd()
    x1 = system.state_dynamics(x=x0, u=np.array([[0]]))
    y1 = system.output_function(x=x1, u=np.array([[0]]))
    assert x1.shape == (8, 1)
    assert y1.shape == (1, 1)


def test_coupled_msd_multiple_carts() -> None:
    for N in range(2, 6):
        N = 2
        system = CoupledMsd(
            N=N,
            k=np.array(range(1, N + 1)),
            m=np.array(range(1, N + 1)),
            c=np.array(range(1, N + 1)),
        )
        x0 = np.zeros(shape=(N * 2, 1))
        x1 = system.state_dynamics(x=x0, u=np.array([[0]]))
        assert x1.shape == (N * 2, 1)


def test_coupled_msd_linearization() -> None:
    system = CoupledMsd()
    A, B = system.get_linearization()
    assert A.shape == (8, 8)
    assert B.shape == (8, 1)


def test_coupled_msd_evaluate_linearization() -> None:
    system = CoupledMsd()
    A_sym, B_sym = system.get_linearization()
    A, B = system.evaluate_linearization(
        A_sym=A_sym,
        B_sym=B_sym,
        x_bar=np.zeros(shape=(8, 1)),
        u_bar=np.array([[0]]),
    )
    assert A.shape == (8, 8)
    assert B.shape == (8, 1)


def test_lure() -> None:
    u = utils.get_input()
    x0 = utils.get_initial_state()
    A, B1, C1, D11 = utils.get_stable_linear_matrices()
    C2 = np.random.normal(size=(4, 2))
    B2 = np.random.normal(size=(2, 4))
    D21 = np.random.normal(size=(4, 1))
    D12 = np.random.normal(size=(1, 4))
    model = Lure(
        A=A,
        B1=B1,
        B2=B2,
        C1=C1,
        C2=C2,
        D11=D11,
        D12=D12,
        D21=D21,
        Delta=np.tanh,
    )
    x1 = model.state_dynamics(x0, u[0])
    y = model.output_layer(xs=[x0, x1], us=[u[0], u[1]])

    assert x1.shape == (2, 1)
    assert y[1].shape == (1, 1)


def test_get_noise() -> None:
    noise_config = NoiseGeneration(type='gaussian', mean=0.0, std=0.01)
    noise_list = get_noise(size=3, lenght=10, config=noise_config)

    assert noise_list[0].shape == (3, 1)
    assert len(noise_list) == 10
