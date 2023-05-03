from typing import Literal, List
from numpy.typing import NDArray
import numpy as np
import dataclasses


@dataclasses.dataclass
class NoiseGeneration:
    type: Literal['gaussian']
    mean: float
    std: float


def get_noise(
    size: int,
    lenght: int,
    config: NoiseGeneration,
) -> List[NDArray[np.float64]]:
    if config.type == 'gaussian':
        noise = [
            noise_element
            for noise_element in np.random.normal(
                loc=config.mean, scale=config.std, size=(lenght, size, 1)
            )
        ]
    else:
        raise NotImplementedError
    return noise
