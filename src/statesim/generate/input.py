from numpy.typing import NDArray
import numpy as np
from typing import List, Optional
from ..configuration import RandomStaticInputConfig, GaussianRandomFieldInputConfig
from sklearn import gaussian_process as gp


def random_static_input(
    N: int,
    nu: int,
    config: RandomStaticInputConfig,
    dt: Optional[float] = None
) -> List[NDArray[np.float64]]:
    """
    Generate a random static input sequence of length N,
    the amplitude is static for a random interval of the range
    `frequency_range` The amplitude of the sequence
    switches after a random number of steps
    in the range of `frequency_range`,
    and stays constant until the next switching point
    or the end of the sequence
    """

    assert config.interval_max < N
    us = np.zeros(shape=(nu, N))
    for element in range(nu):
        k_start = 0
        k_end = 0
        while k_end < N:
            k_end = k_start + int(
                np.random.uniform(
                    low=config.interval_min, high=config.interval_max
                )
            )
            amplitude = np.random.uniform(low=config.u_min, high=config.u_max)
            us[element, k_start:k_end] = amplitude
            k_start = k_end
    return [np.array(u).reshape(nu, 1) for u in us.T]


def gaussian_random_field(
    nu: int,
    N: int, # number of samples in one sequences
    config: GaussianRandomFieldInputConfig,
    dt: Optional[float] = None
) -> List[NDArray]:
    """
    This was used in DeepXDE https://github.com/lululxvi/deepxde and adjusted to meet our needs
    """
    if dt is None:
        ValueError('Step size dt needs to be defined')

    Ker = gp.kernels.RBF(length_scale=config.l)
    x = np.linspace(0, (N-1)*dt, num=N)[:, None]
    Kx = Ker(x)
    L = np.linalg.cholesky(Kx+1e-13*np.eye(N))
    us = config.s * np.random.randn(N,nu)
    vs = np.dot(L,us)
    return [np.array(v).reshape(nu,1) for v in vs]

