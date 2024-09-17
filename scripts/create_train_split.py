import os
import argparse
from pydantic import BaseModel
from statesim.io import get_csv_file_list
from statesim.configuration import SplitConfig, GenerateConfig
import numpy as np
import random
import shutil
from statesim.utils import get_data_directory_name

from pathlib import Path

mode_list = ['train', 'validation', 'test']


def create_train_test_split(
    config: GenerateConfig, raw_data_directory: str
) -> None:
    data_directory = os.path.join(
        os.path.expanduser(raw_data_directory), os.pardir
    )
    processed_directory = os.path.join(
        os.path.join(data_directory, 'processed')
    )
    for mode in mode_list:
        os.makedirs(os.path.join(processed_directory, mode), exist_ok=True)

    csv_files = get_csv_file_list(os.path.expanduser(raw_data_directory))
    n_files = len(csv_files)

    config.split.initial_state = config.simulator.initial_state

    n_train_files = int(config.split.train_split * n_files)
    n_validation_files = int(config.split.validation_split * n_files)
    n_test_files = n_files - n_validation_files - n_train_files

    random.seed = config.split.seed
    files_idx = list(range(n_files))
    random.shuffle(files_idx)

    test_idx = files_idx[:n_test_files]
    validation_idx = files_idx[
        n_test_files : n_test_files + n_validation_files
    ]
    train_idx = files_idx[n_test_files + n_validation_files :]

    config.split.split_filenames = {}
    for mode, mode_idx in zip(
        mode_list, [train_idx, validation_idx, test_idx]
    ):
        mode_csv_files = [csv_files[idx] for idx in mode_idx]
        config.split.split_filenames[mode] = mode_csv_files
        for csv_file in mode_csv_files:
            src = os.path.join(
                os.path.expanduser(raw_data_directory), csv_file
            )
            dst = os.path.join(processed_directory, mode, csv_file)
            shutil.copy(src=src, dst=dst)
            print(f'Copied {src} to {dst}.')

    print('Write configuration file')
    with open(os.path.join(processed_directory, 'config.json'), mode='w') as f:
        f.write(config.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create train, test and validation split of generated data'
    )
    parser.add_argument(
        'system', type=str, help='system name: msd, cartpole, pendulum'
    )

    args = parser.parse_args()
    config_file_path = (
        Path.cwd().joinpath('config').joinpath(f'{args.system}.json')
    )

    config = GenerateConfig.parse_file(config_file_path)
    raw_data_directory = get_data_directory_name(
        Path(os.path.expanduser(config.result_directory)),
        config.base_name,
        config.input_generator.u_max,
        int(config.input_generator.interval_min),
        int(config.input_generator.interval_max),
        config.K,
        int(config.T),
        'raw',
    )
    create_train_test_split(config, raw_data_directory)
