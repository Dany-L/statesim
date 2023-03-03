from ..simulator import SimulationResult
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_states(result: SimulationResult, t: NDArray[np.float64]) -> None:
    nx = result.xs[0].shape[0]

    for element in range(nx):
        plt.plot(t, np.array([x[element] for x in result.xs]))

    plt.show()
