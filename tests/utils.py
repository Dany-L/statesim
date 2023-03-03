from typing import Tuple, List, Callable
from numpy.typing import NDArray
import numpy as np

A = np.array([[0, 1], [0.1, -0.8]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
D = np.array([[0]])


def get_linear_matrices() -> (
    Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]
):
    return (A, B, C, D)


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
