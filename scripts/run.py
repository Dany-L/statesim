from statesim.simulator import ContinuousSimulator
from statesim.io import (
    write_measurement_csv,
    convert_simulation_to_measurement,
)
from statesim.generate.input import generate_random_static_input
from statesim.system.cartpole import CartPole
from statesim.model.statespace import Nonlinear

import os
import time
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from typing import Optional, Union, List


class InputConfig(BaseModel):
    u_min: float
    u_max: float
    interval_min: float
    interval_max: float


class SystemConfig(BaseModel):
    name: str
    A: Optional[List[float]]
    B: Optional[List[float]]


class CartPoleConfig(SystemConfig):
    nx: int
    ny: int
    nu: int
    g: float
    m_c: float
    m_p: float
    length: float
    mu_c: float
    mu_p: float


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
    system: Union[CartPoleConfig, SystemConfig]
    simulator: Optional[SimulatorConfig]


config = GenerateConfig.parse_obj(
    {
        'result_directory': '~/cartpole',
        'folder_name': time.strftime('%Y_%m_%d-%H_%M_%S', time.localtime()),
        'seed': 2023,
        'K': 100,
        'T': 20,
        'step_size': 0.02,
        'input': {
            'u_max': 10,
            'u_min': -10,
            'interval_min': 20,
            'interval_max': 100,
        },
        'system': {
            'name': 'CartPole',
            'nx': 4,
            'ny': 1,
            'nu': 1,
            'g': 9.81,
            'm_c': 1.0,
            'm_p': 0.1,
            'length': 0.5,
            'mu_c': 0.0,
            'mu_p': 0.01,
        },
        'simulator': {'initial_state': [0, 0, 0, 0]},
    }
)

if __name__ == "__main__":
    np.random.seed = config.seed
    sys = CartPole(
        g=config.system.g,
        m_c=config.system.m_c,
        m_p=config.system.m_p,
        length=config.system.length,
        mu_c=config.system.mu_c,
        mu_p=config.system.mu_p,
    )
    A_sym, B_sym = sys.get_linearization()
    A, B = sys.evaluate_linearization(
        A_sym=A_sym,
        B_sym=B_sym,
        x_bar=np.array([[0], [0], [np.pi], [0]]),
        u_bar=np.array([[0]]),
    )
    config.system.A = A.tolist()
    config.system.B = B.tolist()

    sim = ContinuousSimulator(T=config.T, step_size=config.step_size)
    model = Nonlinear(
        f=sys.state_dynamics,
        g=sys.output_function,
        nx=config.system.nx,
        nu=config.system.nu,
        ny=config.system.ny,
    )

    assert os.path.isdir(os.path.expanduser(config.result_directory))
    result_directory_path = os.path.join(
        os.path.expanduser(config.result_directory), config.folder_name
    )
    os.mkdir(result_directory_path)

    print('Write configuration file')
    with open(
        os.path.join(result_directory_path, 'config.json'), mode='w'
    ) as f:
        f.write(config.json())

    N = int(config.T / config.step_size)
    for sample in range(config.K):
        fullfilename = os.path.join(
            result_directory_path,
            f'{sample:04d}_nonlinear_simulation_T_{int(config.T)}.csv',
        )
        us = generate_random_static_input(
            N=N,
            nu=config.system.nu,
            amplitude_range=(config.input.u_min, config.input.u_max),
            frequency_range=[
                config.input.interval_min,
                config.input.interval_max,
            ],
        )
        result, _ = sim.simulate(
            model=model,
            initial_state=np.array(config.simulator.initial_state).reshape(
                config.system.nx, 1
            ),
            input=us,
        )
        print(f'{sample}: write csv file: {fullfilename}')
        write_measurement_csv(
            filepath=fullfilename,
            measure_data=convert_simulation_to_measurement(sim_result=result),
        )
