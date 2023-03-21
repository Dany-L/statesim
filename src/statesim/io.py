from typing import List
from numpy.typing import NDArray
import numpy as np
import os
import pandas as pd
import dataclasses
from .simulator import SimulationResult


@dataclasses.dataclass
class SimulationMeasurement:
    t: NDArray[np.float64]
    ys: List[NDArray[np.float64]]
    us: List[NDArray[np.float64]]


def read_measurement_csv(filepath: str) -> SimulationMeasurement:
    """Read csv file that contains y|y_d, u|u_d and t columns"""
    assert os.path.isfile(filepath)
    df_measure = pd.read_csv(filepath)
    df_y = df_measure.filter(regex='y|y_d')
    df_u = df_measure.filter(regex='u|u_d')
    ny = df_y.shape[1]
    nu = df_u.shape[1]
    return SimulationMeasurement(
        t=np.array(df_measure['t']),
        ys=[np.array(y).reshape(ny, 1) for _, y in df_y.iterrows()],
        us=[np.array(u).reshape(nu, 1) for _, u in df_u.iterrows()],
    )


def write_measurement_csv(
    filepath: str, measure_data: SimulationMeasurement
) -> None:
    """Write input output measurements and time to csv,
    column names are u_<idx>, y_<idx> and t"""
    assert os.path.isdir(os.path.dirname(filepath))
    ny = measure_data.ys[0].shape[0]
    nu = measure_data.us[0].shape[0]
    # initialize datafram
    df = pd.DataFrame()
    loc = 0
    # add time
    df.insert(loc=loc, column='t', value=measure_data.t)
    loc += 1
    # add inputs
    for element in range(nu):
        df.insert(
            loc=loc,
            column=f'u_{element+1}',
            value=[u[element, 0] for u in measure_data.us],
        )
        loc += 1
    # add outputs
    for element in range(ny):
        df.insert(
            loc=loc,
            column=f'y_{element+1}',
            value=[y[element, 0] for y in measure_data.ys],
        )
        loc += 1

    df.to_csv(path_or_buf=filepath, index=False)


def convert_simulation_to_measurement(
    sim_result: SimulationResult,
) -> SimulationMeasurement:
    return SimulationMeasurement(
        t=sim_result.teval, ys=sim_result.ys, us=sim_result.us
    )


def get_csv_file_list(directory: str) -> List[str]:
    assert os.path.isdir(directory)
    csv_files = []
    for f in os.listdir(directory):
        if f.endswith('.csv'):
            csv_files.append(f)
    return csv_files
