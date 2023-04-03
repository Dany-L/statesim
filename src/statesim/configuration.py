from pydantic import BaseModel
from typing import Optional, List, Dict, Union


class SplitConfig(BaseModel):
    raw_data_directory: str
    train_split: float
    validation_split: float
    seed: int
    initial_state: Optional[List]
    split_filenames: Optional[Dict]


class InputConfig(BaseModel):
    u_min: float
    u_max: float
    interval_min: float
    interval_max: float


class SystemConfig(BaseModel):
    name: str
    A: Optional[List]
    B: Optional[List]
    C: List[float]
    xbar: List[float]
    ubar: List[float]
    nx: int
    ny: int
    nu: int


class CartPoleConfig(SystemConfig):
    g: float
    m_c: float
    m_p: float
    length: float
    mu_c: float
    mu_p: float


class PendulumConfig(SystemConfig):
    g: float
    m_p: float
    length: float
    mu_p: float


class CoupledMsdConfig(SystemConfig):
    N: int
    k: List[float]
    c: List[float]
    m: List[float]


class SimulatorConfig(BaseModel):
    initial_state: List[float]
    method: Optional[str]


class GenerateConfig(BaseModel):
    result_directory: str
    folder_name: str
    seed: int
    K: int
    T: float
    step_size: float
    input: InputConfig
    system: Union[CartPoleConfig, PendulumConfig, CoupledMsdConfig]
    simulator: Optional[SimulatorConfig]
