from statesim.model.statespace import Linear, Nonlinear, Lure
from statesim.simulator import ContinuousSimulator, DiscreteSimulator
from statesim.analysis.plot_simulation_results import (
    plot_comparison,
    plot_inputs,
    plot_outputs,
    plot_states,
)
from statesim.noise import get_noise, NoiseGeneration
from statesim.analysis.system_analysis import SystemAnalysisContinuous
from statesim.system.cartpole import CartPole
from statesim.system.inverted_pendulum import InvertedPendulum
from statesim.system.coupled_msd import CoupledMsd
from statesim.io import (
    read_measurement_csv,
    write_measurement_csv,
)
from statesim.configuration import InputGeneratorConfig
from statesim.utils import (
    get_callable_from_method_string,
    get_callable_from_input_config,
    run_simulation_write_csv_files,
)
from statesim.generate.input import random_static_input
from typing import List, Dict, Callable
import utils
import numpy as np
import sympy as sym
import os
import math
import pathlib


def test_plot_states() -> None:
    results = utils.get_simulation_results()
    plot_states(result=results[0])


def test_plot_inputs() -> None:
    results = utils.get_simulation_results()
    plot_inputs(result=results[0])


def test_plot_outputs() -> None:
    results = utils.get_simulation_results()
    plot_outputs(result=results[0])


def test_write_measurement_csv() -> None:
    data = utils.get_measurement_data()
    filepath = os.path.join(utils.get_tmp_directory(), 'measurement.csv')
    write_measurement_csv(filepath=filepath, simulation_data=data)


def test_plot_magnitude() -> None:
    ana = SystemAnalysisContinuous(utils.get_stable_linear_matrices())
    ana.plot_magnitude()


def test_get_peak_gain() -> None:
    ana = SystemAnalysisContinuous(utils.get_stable_linear_matrices())
    ana.get_peak_gain()


def test_get_real_eigenvalues() -> None:
    ana = SystemAnalysisContinuous(utils.get_stable_linear_matrices())
    ana.get_real_eigenvalues()


def test_analysis() -> None:
    ana = SystemAnalysisContinuous(utils.get_stable_linear_matrices())
    ana.analysis()
    ana = SystemAnalysisContinuous(utils.get_unstable_linear_matrices())
    ana.analysis()


def test_run_simulation_write_csv_files(tmp_path: pathlib.Path) -> None:
    generate_config = utils.get_generate_config(str(tmp_path))
    model = Linear(
        A=np.array(generate_config.system.A),
        B=np.array(generate_config.system.B),
        C=np.array(generate_config.system.C),
        D=np.array(generate_config.system.D),
    )
    generate_config.system.nu = len(generate_config.system.B[0])
    generate_config.system.nx = len(generate_config.system.A)
    generate_config.system.ny = len(generate_config.system.C)
    sim = ContinuousSimulator(
        T=generate_config.T,
        step_size=generate_config.step_size,
    )
    input_generator = get_callable_from_input_config(
        generate_config.input_generator
    )
    run_simulation_write_csv_files(
        config=generate_config,
        model=model,
        sim=sim,
        result_directory_path=tmp_path,
        input_generator=input_generator,
    )
