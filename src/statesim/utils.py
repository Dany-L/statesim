from .configuration import GenerateConfig, CSV_FILE_NAME, InputGeneratorConfig
from .model.statespace import StateSpaceModel
from .io import write_measurement_csv
from .simulator import BasicSimulator
from .generate.input import random_static_input, gaussian_random_field
from .analysis import plot_simulation_results as plot
import pathlib
import os
import numpy as np
from typing import Callable, Optional, Union
import matplotlib.pyplot as plt


__all__ = ('random_static_input','gaussian_random_field')


def get_data_directory_name(
    root_directory: pathlib.Path,
    base_name: str,
    sequence_count: int,
    sequence_length: int,
    raw_directory_name: str,
) -> pathlib.Path:
    return root_directory.joinpath(
        f'{base_name}_M-{sequence_count}_T-{sequence_length}'
    ).joinpath(raw_directory_name)


def run_simulation_write_csv_files(
    config: GenerateConfig,
    model: StateSpaceModel,
    sim: BasicSimulator,
    result_directory_path: pathlib.Path,
    input_generator: Optional[Callable] = None,
) -> None:
    N = int(config.T / config.step_size)
    for sample in range(config.M):
        fullfilename = os.path.join(
            result_directory_path,
            f'{sample:04d}_{CSV_FILE_NAME}_T_{int(config.T)}.csv',
        )
        if input_generator is not None:
            us = input_generator(
                N=N, nu=config.system.nu, config=config.input_generator, dt=config.step_size
            )
        else:
            us = [
                u.reshape((config.system.nu, 1))
                for u in np.zeros(
                    shape=(N, config.system.nu), dtype=np.float64
                )
            ]
        result = sim.simulate(
            model=model,
            initial_state=np.array(config.simulator.initial_state).reshape(
                config.system.nx, 1
            ),
            input=us,
            noise_config=config.measurement_noise,
        )
        # if sample == 0:
            # plot.plot_outputs(result)
            # plot.plot_inputs(result)
            # plt.show()

        print(f'{sample}: write csv file: {fullfilename}')
        write_measurement_csv(
            filepath=fullfilename,
            simulation_data=result,
        )


def get_callable_from_input_config(
    config: Union[InputGeneratorConfig, None]
) -> Union[None, Callable]:
    return (
        None
        if config is None
        else get_callable_from_method_string(config.type)
    )


def get_callable_from_method_string(
    method: Union[str, None]
) -> Union[None, Callable]:
    return None if method is None else eval(method)
