from statesim.model.statespace import (
    Linear,
    Nonlinear,
)
from statesim.simulator import ContinuousSimulator, DiscreteSimulator
from statesim.plot.plot_simulation_results import plot_comparison
from statesim.system.cartpole import CartPole
from typing import List, Dict
import utils
import numpy as np
import sympy as sym
from statesim.io import read_measurement_csv
import os

DIRNAME = os.path.dirname(__file__)


def test_cartpole_compare_simulation_results_continuous_linear() -> None:
    filepath = os.path.join(
        DIRNAME, 'data/2023_03_09-01_25_33_cartpole_linear_continous.csv'
    )
    read_measurement_csv(filepath=filepath)
    assert False


def test_cartpole_compare_simulation_results_continuous_nonlinear() -> None:
    pass


def test_cartpole_compare_simulation_results_discrete_linear() -> None:
    pass
