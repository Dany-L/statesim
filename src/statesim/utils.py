from .configuration import GenerateConfig, CSV_FILE_NAME, InputGeneratorConfig
from .model.statespace import StateSpaceModel
from .io import write_measurement_csv
from .simulator import BasicSimulator
from .generate.input import random_static_input
import pathlib
import os
import numpy as np
from typing import Callable, Optional, Union


__all__ = ('random_static_input',)


def run_simulation_write_csv_files(
    config: GenerateConfig,
    model: StateSpaceModel,
    sim: BasicSimulator,
    result_directory_path: pathlib.Path,
    input_generator: Optional[Callable] = None,
) -> None:
    N = int(config.T / config.step_size)
    for sample in range(config.K):
        fullfilename = os.path.join(
            result_directory_path,
            f'{sample:04d}_{CSV_FILE_NAME}_T_{int(config.T)}.csv',
        )
        if input_generator is not None:
            us = input_generator(
                N=N, nu=config.system.nu, config=config.input_generator
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
        #     plot_outputs(result)
        #     plot_inputs(result)
        #     plt.show()

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
