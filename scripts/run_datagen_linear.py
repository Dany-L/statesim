from statesim.simulator import ContinuousSimulator
from statesim.model.statespace import Linear
from statesim.configuration import (
    GenerateConfig,
)
from statesim.analysis.plot_simulation_results import plot_outputs
from statesim.utils import (
    run_simulation_write_csv_files,
    get_callable_from_input_config,
)
from statesim.generate.input import random_static_input

import os
import pathlib
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def main(config_file: GenerateConfig) -> None:
    config = GenerateConfig.parse_file(config_file)
    model = Linear(
        A=np.array(config.system.A),
        B=np.array(config.system.B),
        C=np.array(config.system.C),
        D=np.array(config.system.D),
    )
    config.system.nu = len(config.system.B[0])
    config.system.nx = len(config.system.A)
    config.system.ny = len(config.system.C)

    sim = ContinuousSimulator(T=config.T, step_size=config.step_size)

    result_directory_path = os.path.join(
        os.path.expanduser(config.result_directory),
        f'{config.base_name}_K-{config.K}_T-{int(config.T)}',
        'raw',
    )

    os.makedirs(result_directory_path, exist_ok=True)
    run_simulation_write_csv_files(
        config=config,
        model=model,
        sim=sim,
        result_directory_path=result_directory_path,
        input_generator=get_callable_from_input_config(config.input_generator),
    )


if __name__ == "__main__":
    config_file_path = (
        pathlib.Path.cwd().joinpath('config').joinpath('linear.json')
    )
    main(config_file=config_file_path)
