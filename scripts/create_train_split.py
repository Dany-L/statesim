import os
from pydantic import BaseModel
from statesim.io import get_csv_file_list
from statesim.configuration import SplitConfig, GenerateConfig
import numpy as np
import random
import shutil

from pathlib import Path

config = SplitConfig.parse_obj(
    {
        'raw_data_directory': '/Users/jack/pendulum/initial_state_0_K-100_T-20/raw',
        'train_split': 0.6,
        'validation_split': 0.1,
        'seed': 2023,
    }
)

mode_list = ['train', 'validation', 'test']

if __name__ == "__main__":
    data_directory = os.path.join(
        os.path.expanduser(config.raw_data_directory), os.pardir
    )
    processed_directory = os.path.join(
        os.path.join(data_directory, 'processed')
    )
    for mode in mode_list:
        os.makedirs(os.path.join(processed_directory, mode), exist_ok=True)

    csv_files = get_csv_file_list(
        os.path.expanduser(config.raw_data_directory)
    )
    n_files = len(csv_files)

    raw_config = GenerateConfig.parse_file(
        path=Path(
            os.path.join(
                os.path.expanduser(config.raw_data_directory), 'config.json'
            )
        )
    )
    config.initial_state = raw_config.simulator.initial_state

    n_train_files = int(config.train_split * n_files)
    n_validation_files = int(config.validation_split * n_files)
    n_test_files = n_files - n_validation_files - n_train_files

    random.seed = config.seed
    files_idx = list(range(n_files))
    random.shuffle(files_idx)

    test_idx = files_idx[:n_test_files]
    validation_idx = files_idx[
        n_test_files : n_test_files + n_validation_files
    ]
    train_idx = files_idx[n_test_files + n_validation_files :]

    config.split_filenames = {}
    for mode, mode_idx in zip(
        mode_list, [train_idx, validation_idx, test_idx]
    ):
        mode_csv_files = [csv_files[idx] for idx in mode_idx]
        config.split_filenames[mode] = mode_csv_files
        for csv_file in mode_csv_files:
            src = os.path.join(
                os.path.expanduser(config.raw_data_directory), csv_file
            )
            dst = os.path.join(processed_directory, mode, csv_file)
            shutil.copy(src=src, dst=dst)
            print(f'Copied {src} to {dst}.')

    print('Write configuration file')
    with open(os.path.join(processed_directory, 'config.json'), mode='w') as f:
        f.write(config.json())
