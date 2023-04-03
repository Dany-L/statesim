from statesim.simulator import ContinuousSimulator
from statesim.io import (
    write_measurement_csv,
)
from statesim.generate.input import generate_random_static_input
from statesim.system.cartpole import CartPole
from statesim.system.coupled_msd import CoupledMsd
from statesim.system.inverted_pendulum import InvertedPendulum
from statesim.model.statespace import Nonlinear
from statesim.configuration import (
    GenerateConfig,
    CartPoleConfig,
    CoupledMsdConfig,
    PendulumConfig,
)

import os
import time
import numpy as np
from numpy.typing import NDArray

# config = GenerateConfig.parse_obj(
#     {
#         'result_directory': '~/cartpole',
#         'folder_name': 'initial_state_0_K_100_T_20_u_static_random',
#         'seed': 2023,
#         'K': 100,
#         'T': 20,
#         'step_size': 0.02,
#         'input': {
#             'u_max': 10,
#             'u_min': -10,
#             'interval_min': 20,
#             'interval_max': 100,
#         },
#         'system': {
#             'name': 'CartPole',
#             'nx': 4,
#             'ny': 1,
#             'nu': 1,
#             'C': [0.0, 0.0, 1.0, 0.0],
#             'xbar': [0.0, 0.0, np.pi, 0.0],
#             'ubar': [0.0],
#             'g': 9.81,
#             'm_c': 1.0,
#             'm_p': 0.1,
#             'length': 0.5,
#             'mu_c': 0.0,
#             'mu_p': 0.01,
#         },
#         'simulator': {'initial_state': [0, 0, 0, 0]},
#     }
# )
# config = GenerateConfig.parse_obj(
#     {
#         'result_directory': '~/mass-spring-damper',
#         'folder_name': 'initial_state_0_K_200_T_30_u_static_random',
#         'seed': 2023,
#         'K': 200,
#         'T': 30,
#         'step_size': 0.05,
#         'input': {
#             'u_max': 4,
#             'u_min': -4,
#             'interval_min': 20,
#             'interval_max': 100,
#         },
#         'system': {
#             'name': 'CoupledMsd',
#             'nx': 8,
#             'ny': 1,
#             'nu': 1,
#             'C': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
#             'xbar': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#             'ubar': [0.0],
#             'N': 4,
#             'k': [0.25, 0.33, 0.4167, 0.5],
#             'c': [0.25, 0.33, 0.4167, 0.5],
#             'm': [1.0, 0.833, 0.67, 0.5]
#         },
#         'simulator': {'initial_state': [0, 0, 0, 0, 0, 0, 0, 0]},
#     }
# )
config = GenerateConfig.parse_obj(
    {
        'result_directory': '~/pendulum',
        'folder_name': 'initial_state_0_K_100_T_20_u_static_random',
        'seed': 2023,
        'K': 100,
        'T': 20,
        'step_size': 0.02,
        'input': {
            'u_max': 2,
            'u_min': -2,
            'interval_min': 20,
            'interval_max': 100,
        },
        'system': {
            'name': 'Pendulum',
            'nx': 2,
            'ny': 1,
            'nu': 1,
            'C': [1.0, 0.0],
            'xbar': [np.pi, 0.0],
            'ubar': [0.0],
            'g': 9.81,
            'm_p': 0.1,
            'length': 0.5,
            'mu_p': 0.01,
        },
        'simulator': {'initial_state': [0, 0]},
    }
)


if __name__ == "__main__":
    np.random.seed = config.seed
    if isinstance(config.system, CartPoleConfig):
        sys = CartPole(
            g=config.system.g,
            m_c=config.system.m_c,
            m_p=config.system.m_p,
            length=config.system.length,
            mu_c=config.system.mu_c,
            mu_p=config.system.mu_p,
        )
    elif isinstance(config.system, CoupledMsdConfig):
        sys = CoupledMsd(
            N=config.system.N,
            c=config.system.c,
            k=config.system.k,
            m=config.system.m,
        )
    elif isinstance(config.system, PendulumConfig):
        sys = InvertedPendulum(
            g=config.system.g,
            m_p=config.system.m_p,
            length=config.system.length,
            mu_p=config.system.mu_p,
        )
    else:
        raise NotImplementedError
    A_sym, B_sym = sys.get_linearization()
    A, B = sys.evaluate_linearization(
        A_sym=A_sym,
        B_sym=B_sym,
        x_bar=np.array(config.system.xbar, dtype=np.float64).reshape(
            (config.system.nx, 1)
        ),
        u_bar=np.array(config.system.ubar, dtype=np.float64).reshape(
            (config.system.nu, 1)
        ),
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
        os.path.expanduser(config.result_directory), config.folder_name, 'raw'
    )
    os.makedirs(result_directory_path, exist_ok=True)

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
            simulation_data=result,
        )
