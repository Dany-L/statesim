from typing import List
import numpy as np
import os
import pandas as pd
from .simulator import SimulationData


def read_measurement_csv(filepath: str) -> SimulationData:
    """Read csv file that contains y|y_d, u|u_d and t columns"""
    assert os.path.isfile(filepath)
    df_measure = pd.read_csv(filepath)
    df_y = df_measure.filter(regex='y|y_d')
    df_u = df_measure.filter(regex='u|u_d')
    ny = df_y.shape[1]
    nu = df_u.shape[1]
    return SimulationData(
        t=np.array(df_measure['t']),
        ys=[np.array(y).reshape(ny, 1) for _, y in df_y.iterrows()],
        us=[np.array(u).reshape(nu, 1) for _, u in df_u.iterrows()],
        xs=[],
        name='unknown',
    )


def write_measurement_csv(
    filepath: str, simulation_data: SimulationData
) -> None:
    """Write input output measurements and time to csv,
    column names are u_<idx>, y_<idx> and t"""
    assert os.path.isdir(os.path.dirname(filepath))
    ny = simulation_data.ys[0].shape[0]
    nu = simulation_data.us[0].shape[0]
    # initialize datafram
    df = pd.DataFrame()
    loc = 0
    # add time
    df.insert(loc=loc, column='t', value=simulation_data.t)
    loc += 1
    # add inputs
    for element in range(nu):
        df.insert(
            loc=loc,
            column=f'u_{element+1}',
            value=[u[element, 0] for u in simulation_data.us],
        )
        loc += 1
    # add outputs
    for element in range(ny):
        df.insert(
            loc=loc,
            column=f'y_{element+1}',
            value=[y[element, 0] for y in simulation_data.ys],
        )
        loc += 1
    # add states
    if simulation_data.xs:
        nx = simulation_data.xs[0].shape[0]
        for element in range(nx):
            df.insert(
                loc=loc,
                column=f'x_{element+1}',
                value=[x[element, 0] for x in simulation_data.xs],
            )
            loc += 1

    df.to_csv(path_or_buf=filepath, index=False)


def get_csv_file_list(directory: str) -> List[str]:
    assert os.path.isdir(directory)
    csv_files = []
    for f in os.listdir(directory):
        if f.endswith('.csv'):
            csv_files.append(f)
    return csv_files
