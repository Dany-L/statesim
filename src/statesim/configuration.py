from pydantic import BaseModel
from typing import Optional, List, Dict, Union, Literal


CSV_FILE_NAME = 'simulation'


class SplitConfig(BaseModel):
    train_split: float
    validation_split: float
    seed: int
    initial_state: Optional[List]
    split_filenames: Optional[Dict]


class InputGeneratorConfig(BaseModel):
    type: Literal['random_static_input', 'np.sin', 'gaussian_random_field']

class RandomStaticInputConfig(InputGeneratorConfig):
    u_min: float
    u_max: float
    interval_min: float
    interval_max: float

class GaussianRandomFieldInputConfig(InputGeneratorConfig):
    l: float
    s: float


class LinearSystemConfig(BaseModel):
    name: str
    A: List[List[float]]
    B: List[List[float]]
    C: List[List[float]]
    D: List[List[float]]
    nx: Optional[int]
    ny: Optional[int]
    nu: Optional[int]


class NonlinearSystemConfig(BaseModel):
    name: str
    A: Optional[List[List[float]]]
    B: Optional[List[List[float]]]
    C: List[List[float]]
    xbar: List[float]
    ubar: List[float]
    nx: int
    ny: int
    nu: int


class CartPoleConfig(NonlinearSystemConfig):
    g: float
    m_c: float
    m_p: float
    length: float
    mu_c: float
    mu_p: float


class PendulumConfig(NonlinearSystemConfig):
    g: float
    m_p: float
    length: float
    mu_p: float


class CoupledMsdConfig(NonlinearSystemConfig):
    N: int
    k: List[float]
    c: List[float]
    m: List[float]


class SimulatorConfig(BaseModel):
    initial_state: List[float]
    method: Optional[str]


class NoiseConfig(BaseModel):
    type: Literal['gaussian']
    mean: float
    std: float


class GenerateConfig(BaseModel):
    result_directory: str
    base_name: str
    seed: int
    M: int
    T: float
    step_size: float
    input_generator: Optional[Union[GaussianRandomFieldInputConfig, RandomStaticInputConfig]]
    system: Union[
        LinearSystemConfig, CartPoleConfig, PendulumConfig, CoupledMsdConfig
    ]
    simulator: Optional[SimulatorConfig]
    measurement_noise: Optional[NoiseConfig]
    split: SplitConfig
