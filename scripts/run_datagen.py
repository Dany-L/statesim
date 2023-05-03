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
from statesim.analysis.plot_simulation_results import plot_outputs

import os
import argparse
import pathlib
import time
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def main(config_file: pathlib.Path):
    config = GenerateConfig.parse_file(config_file)
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
    config.system.A = (A * config.step_size + np.eye(A.shape[0])).tolist()
    config.system.B = (B * config.step_size).tolist()

    sim = ContinuousSimulator(T=config.T, step_size=config.step_size)
    model = Nonlinear(
        f=sys.state_dynamics,
        g=sys.output_function,
        nx=config.system.nx,
        nu=config.system.nu,
        ny=config.system.ny,
    )

    # assert os.path.isdir(os.path.expanduser(config.result_directory))
    result_directory_path = os.path.join(
        os.path.expanduser(config.result_directory),
        f'{config.base_name}_K-{config.K}_T-{int(config.T)}',
        'raw',
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
        result = sim.simulate(
            model=model,
            initial_state=np.array(config.simulator.initial_state).reshape(
                config.system.nx, 1
            ),
            input=us,
            noise_config=config.measurement_noise,
        )
        if sample == 1:
            plot_outputs(result)
            plt.show()

        print(f'{sample}: write csv file: {fullfilename}')
        write_measurement_csv(
            filepath=fullfilename,
            simulation_data=result,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Simulate data for dynamical systems'
    )
    parser.add_argument(
        'system', type=str, help='system name: msd, cartpole, pendulum'
    )

    args = parser.parse_args()
    config_file_path = (
        pathlib.Path.cwd().joinpath('config').joinpath(f'{args.system}.json')
    )
    main(config_file=config_file_path)
