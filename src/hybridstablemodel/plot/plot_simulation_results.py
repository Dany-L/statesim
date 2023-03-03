from ..simulator import SimulationResult
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

plt.rcParams['text.usetex'] = True

def plot_states(result: SimulationResult, t: NDArray[np.float64]) -> None:
    nx = result.xs[0].shape[0]
    fig, axs = plt.subplots(nrows=nx, ncols=1, tight_layout=True, squeeze=False)
    fig.suptitle('State plots')
    for element, ax in zip(range(nx), axs[:, 0]):
        ax.plot(t, np.array([x[element] for x in result.xs]))
        ax.set_title(f'$x_{element+1}$')
        ax.grid()

def plot_inputs(result: SimulationResult, t: NDArray[np.float64]) -> None:
    nu = result.us[0].shape[0]
    fig, axs = plt.subplots(nrows=nu, ncols=1, tight_layout=True, squeeze=False)
    fig.suptitle('Input plots')
    for element, ax in zip(range(nu), axs[:, 0]):
        u = np.array([u[element] for u in result.us])
        ax.plot(t, np.array([u[element] for u in result.us]))
        ax.set_title(f'$u_{element+1}$')
        ax.grid()
    

def plot_outputs(result: SimulationResult, t: NDArray[np.float64]) -> None:
    ny = result.ys[0].shape[0]
    fig, axs = plt.subplots(nrows=ny, ncols=1, tight_layout=True, squeeze=False)
    fig.suptitle('Output plots')
    for element, ax in zip(range(ny), axs[:, 0]):
        ax.plot(t, np.array([y[element] for y in result.ys]))
        ax.set_title(f'$y_{element+1}$')
        ax.grid()