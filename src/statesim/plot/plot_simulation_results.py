from ..simulator import SimulationResult
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Literal

plt.rcParams['text.usetex'] = True


def plot_states(result: SimulationResult) -> None:
    nx = result.xs[0].shape[0]
    fig, axs = plt.subplots(
        nrows=nx, ncols=1, tight_layout=True, squeeze=False
    )
    fig.suptitle('State plots')
    for element, ax in zip(range(nx), axs[:, 0]):
        ax.plot(result.teval, np.array([x[element] for x in result.xs]))
        ax.set_title(f'$x_{element+1}$')
        ax.grid()


def plot_inputs(result: SimulationResult) -> None:
    nu = result.us[0].shape[0]
    fig, axs = plt.subplots(
        nrows=nu, ncols=1, tight_layout=True, squeeze=False
    )
    fig.suptitle('Input plots')
    for element, ax in zip(range(nu), axs[:, 0]):
        ax.plot(result.teval, np.array([u[element] for u in result.us]))
        ax.set_title(f'$u_{element+1}$')
        ax.grid()


def plot_outputs(result: SimulationResult) -> None:
    ny = result.ys[0].shape[0]
    fig, axs = plt.subplots(
        nrows=ny, ncols=1, tight_layout=True, squeeze=False
    )
    fig.suptitle('Output plots')
    for element, ax in zip(range(ny), axs[:, 0]):
        ax.plot(result.teval, np.array([y[element] for y in result.ys]))
        ax.set_title(f'$y_{element+1}$')
        ax.grid()


def plot_comparison(
    results: List[SimulationResult], type: Literal['xs', 'us', 'ys']
) -> None:
    n = getattr(results[0], type)[0].shape[0]
    fig, axs = plt.subplots(nrows=n, ncols=1, tight_layout=True, squeeze=False)
    fig.suptitle(f'{type} plots')
    for element, ax in zip(range(n), axs[:, 0]):
        for result in results:
            (line,) = ax.plot(
                result.teval,
                np.array([x[element] for x in getattr(result, type)]),
            )
            line.set_label(f'{result.name}')
        ax.set_title(f'${type[:-1]}_{element+1}$')
        ax.legend()
        ax.grid()
    if len(results) == 2:
        # print error
        fig, axs = plt.subplots(
            nrows=n, ncols=1, tight_layout=True, squeeze=False
        )
        fig.suptitle(f'Absolute error {type}')
        for element, ax in zip(range(n), axs[:, 0]):
            (line,) = ax.plot(
                result.teval,
                np.array(
                    [
                        np.abs(x1[element] - x2[element])
                        for x1, x2 in zip(
                            getattr(results[0], type),
                            getattr(results[1], type),
                        )
                    ]
                ),
            )
            ax.set_title(f'${type[:-1]}_{element+1}$')
            ax.legend()
            ax.grid()
