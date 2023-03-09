from typing import List
from numpy.typing import NDArray
import numpy as np
import os
import pandas as pd
import dataclasses

DIRNAME = os.path.dirname(__file__)


@dataclasses.dataclass
class SimulationMeasurement:
    t: NDArray[np.float64]
    ys: List[NDArray[np.float64]]
    us: List[NDArray[np.float64]]


def read_measurement_csv(filepath: str) -> SimulationMeasurement:
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
