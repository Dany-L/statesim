from typing import Tuple, List, Callable
from numpy.typing import NDArray
import numpy as np
from statesim.simulator import SimulationData
from statesim.io import SimulationData
from statesim.configuration import (
    GenerateConfig,
    InputGeneratorConfig,
    LinearSystemConfig,
    SimulatorConfig,
    NoiseConfig,
)
import os

DIRNAME = os.path.dirname(__file__)

A = np.array([[0, 1], [-1, -5]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
D = np.array([[0]])

A_un = np.array([[0, 1], [0.1, -0.8]])
B_un = np.array([[0], [1]])
C_un = np.array([[1, 0]])
D_un = np.array([[0]])


def get_generate_config(result_directory: str) -> GenerateConfig:
    return GenerateConfig(
        result_directory=result_directory,
        base_name='test',
        seed=2023,
        K=10,
        T=5.0,
        step_size=0.01,
        input_generator=InputGeneratorConfig(
            type='random_static_input',
            u_min=-1.0,
            u_max=1.0,
            interval_min=10,
            interval_max=20,
        ),
        system=LinearSystemConfig(
            name='testSys',
            A=A.tolist(),
            B=B.tolist(),
            C=C.tolist(),
            D=D.tolist(),
        ),
        simulator=SimulatorConfig(initial_state=[1.0, 0.0]),
        measurement_noise=NoiseConfig(type='gaussian', mean=0.0, std=0.01),
    )


def get_tmp_directory() -> str:
    return os.path.join(DIRNAME, '_tmp')


def get_directory() -> str:
    return DIRNAME


def get_stable_linear_matrices() -> (
    Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]
):
    return (A, B, C, D)


def get_unstable_linear_matrices() -> (
    Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]
):
    return (A_un, B_un, C_un, D_un)


def get_input() -> List[NDArray[np.float64]]:
    return [
        np.array([[-1]]),
        np.array([[0]]),
        np.array([[0.4]]),
        np.array([[0.9]]),
    ]


def get_initial_state() -> NDArray[np.float64]:
    return np.array([[0.5], [-0.9]])


def get_nonlinear_state_function() -> Callable:
    def state_function(x, u):
        x1_dot = np.squeeze(x[1])
        x2_dot = np.squeeze(x[0] ** 2 + np.sin(x[1]) + 0.5 * u)
        return np.array([[x1_dot], [x2_dot]])

    return state_function


def get_nonlinear_output_function() -> Callable:
    def output_function(x, u):
        return np.array([[1, 0]]) @ x

    return output_function


def get_state_size() -> int:
    return int(get_initial_state().shape[0])


def get_input_size() -> int:
    return int(get_input()[0].shape[0])


def get_output_size() -> int:
    return int(C.shape[0])


def get_simulation_results() -> List[SimulationData]:
    N = 10
    nx = 4
    ny = 2
    nu = 3
    return [
        SimulationData(
            xs=[
                x.reshape(nx, 1)
                for x in np.random.standard_normal(size=(N, nx))
            ],
            us=[
                u.reshape(nu, 1)
                for u in np.random.standard_normal(size=(N, nu))
            ],
            ys=[
                y.reshape(ny, 1)
                for y in np.random.standard_normal(size=(N, ny))
            ],
            t=np.linspace(0, 9, N),
            name='one',
        ),
        SimulationData(
            xs=[
                x.reshape(nx, 1)
                for x in np.random.standard_normal(size=(N, nx))
            ],
            us=[
                u.reshape(nu, 1)
                for u in np.random.standard_normal(size=(N, nu))
            ],
            ys=[
                y.reshape(ny, 1)
                for y in np.random.standard_normal(size=(N, ny))
            ],
            t=np.linspace(0, 9, N),
            name='two',
        ),
    ]


def get_linearization_point_cartpole() -> NDArray[np.float64]:
    return np.array([[0], [0], [np.pi], [0]])


def get_linearization_point_inverted_pendulum() -> NDArray[np.float64]:
    return np.array([[np.pi], [0]])


def get_initial_state_cartpole() -> NDArray[np.float64]:
    return np.array([[0], [0], [np.pi + 0.1], [0]])


def get_initial_state_inverted_pendulum() -> NDArray[np.float64]:
    return np.array([[np.pi + 0.1], [0]])


def get_initial_state_msd() -> NDArray[np.float64]:
    return np.array([[0], [0], [0], [0], [0], [0], [0], [0]])


def calculate_error(
    ys: List[NDArray[np.float64]], ys_hat: List[NDArray[np.float64]]
) -> np.float64:
    assert len(ys) == len(ys_hat)
    error = 0
    T = len(ys)
    for y, y_hat in zip(ys, ys_hat):
        error += np.linalg.norm(y - y_hat, ord=2)
    return error / T


def get_measurement_data() -> SimulationData:
    T = 4
    eta = 0.2
    N = int(T / eta)
    ny = 2
    nu = 3
    return SimulationData(
        t=np.linspace(0, T - eta, N),
        ys=[y.reshape(ny, 1) for y in np.random.standard_normal(size=(N, ny))],
        us=[u.reshape(nu, 1) for u in np.random.standard_normal(size=(N, nu))],
        xs=[],
        name='unknown',
    )
