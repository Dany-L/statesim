from numpy.typing import NDArray
import numpy as np
from typing import List, Tuple


def generate_random_static_input(
    N: int,
    nu: int,
    amplitude_range: Tuple[float, float],
    frequency_range: Tuple[int, int],
) -> List[NDArray[np.float64]]:
    """
    Generate a random static input sequence of length N, the amplitude is static for a random interval of the range `frequency_range`

    The amplitude of the sequence switches after a random number of steps in the range of `frequency_range`, and stays constant until the next switching point or the end of the sequence
    """

    assert frequency_range[1] < N
    us = np.zeros(shape=(nu, N))
    for element in range(nu):
        k_start = 0
        k_end = 0
        while k_end < N:
            k_end = k_start + int(
                np.random.uniform(
                    low=frequency_range[0], high=frequency_range[1]
                )
            )
            amplitude = np.random.uniform(
                low=amplitude_range[0], high=amplitude_range[1]
            )
            us[element, k_start:k_end] = amplitude
            k_start = k_end
    return [np.array(u).reshape(nu, 1) for u in us.T]
