from statesim.simulator import ContinuousSimulator
from statesim.model.statespace import Linear, Nonlinear
from statesim.configuration import (
    GenerateConfig,
)
from statesim.system.cartpole import CartPole
from statesim.system.coupled_msd import CoupledMsd
from statesim.system.pendulum import InvertedPendulum, ActuatedPendulum
from statesim.configuration import (
    GenerateConfig,
    CartPoleConfig,
    CoupledMsdConfig,
    PendulumConfig,
)
from statesim.analysis import plot_simulation_results as plot
from statesim.utils import (
    run_simulation_write_csv_files,
    get_callable_from_input_config,
)

import os
import argparse
import pathlib
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def main(config_file: GenerateConfig) -> None:
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
        if 'Actuated' in config.system.name:
            sys = ActuatedPendulum(
                g=config.system.g,
                m_p=config.system.m_p,
                length=config.system.length,
                mu_p=config.system.mu_p,
            )
        elif 'Inverted' in config.system.name:
            sys = InvertedPendulum(
                g=config.system.g,
                m_p=config.system.m_p,
                length=config.system.length,
                mu_p=config.system.mu_p,
            )
        else:
            raise NotImplementedError
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
    model = Linear(
        A,B,np.array([[1.0,0]]), np.array([[0]])
    )
    model_nl = Nonlinear(
        sys.state_dynamics,
        sys.output_function,
        sys.nx,
        sys.ny,
        sys.nu
    )
    result_directory_path = os.path.join(
        os.path.expanduser(config.result_directory),
        f'{config.base_name}_M-{config.M}_T-{int(config.T)}',
        'raw',
    )
    input_generator=get_callable_from_input_config(config.input_generator)
    N = int(config.T / config.step_size)
    us = input_generator(
        N=N, nu=config.system.nu, config=config.input_generator, dt=config.step_size
    )
    result_lin = sim.simulate(
        model=model,
        initial_state=np.array(config.simulator.initial_state).reshape(
            config.system.nx, 1
        ),
        input=us,
        noise_config=config.measurement_noise,
        name='linear'
    )
    result_nonlin = sim.simulate(
        model=model_nl,
        initial_state=np.array(config.simulator.initial_state).reshape(
            config.system.nx, 1
        ),
        input=us,
        noise_config=config.measurement_noise,
        name='nonlin'
    )
    plot.plot_comparison([result_lin,result_nonlin], 'ys')
    plt.show()

    os.makedirs(result_directory_path, exist_ok=True)

    print('Write configuration file')
    with open(
        os.path.join(result_directory_path, 'config.json'), mode='w'
    ) as f:
        f.write(config.json())
    run_simulation_write_csv_files(
        config=config,
        model=model,
        sim=sim,
        result_directory_path=result_directory_path,
        input_generator=get_callable_from_input_config(config.input_generator),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Simulate data for dynamical systems'
    )
    parser.add_argument(
        'system', type=str, help='system name: msd, cartpole, actuated_pendulum, inverted_pendulum'
    )

    args = parser.parse_args()
    config_file_path = (
        pathlib.Path.cwd().joinpath('config').joinpath(f'{args.system}.json')
    )
    main(config_file=config_file_path)
