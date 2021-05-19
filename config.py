# -*- coding: utf-8 -*-
import os
import re
import glob
import itertools
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
# sacred.SETTINGS.CAPTURE_MODE = 'no' # control the info output
ex = Experiment('InterGI')
ex.captured_out_filter = apply_backspaces_and_linefeeds
source_folders = ['.', './dataload', './models', './utils']  # where the source code placed
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))  # snap all python file
for source_file in sources_to_save:
    ex.add_source_file(source_file)


@ex.config
def cfg():
    isdebug = False
    # isdebug = True

    step = "Class"  # Seg or Class
    mode = "train"  # train or pre_trian

    dataset = "EndoCV"
    seed = 1218
    if step == 'Seg':
        model = "ResegNet"  # ResegNet, SUMNet,
        gpu_devices = 0
        epochs = 200
        batch_size = 4
        if mode == "train":
            optima = {
                'lr': 1e-3,
                'step_size': 20,
                'gamma': 0.1,
                'momentum': 0.9,
                'weight_decay': 0.0005
            }
        elif mode == "test":
            snapshot = './run/InterGI_[train]/1/snapshot/best.pth'
        exp_str = '_'.join([dataset, ] + [f'_[{mode}]'])
        path = {
            'log_dir': './runs',
            'EndoCV': {'data_dir': '../../data/EndoCV2020'},
        }
    elif step == 'Class':
        model = "CasSeg"  # CasSeg,
        gpu_devices = 0
        epochs = 200
        batch_size = 8
        optima = {
            'lr': 1e-4,
            'step_size': 20,
            'gamma': 0.1,
            'momentum': 0.9,
            'weight_decay': 0.005
        }
        if mode == "pre_trian":
            pre_pth_path = './runs/InterGI_ResegNet_EndoCV__[train]/ResegNet_mean_best.pt'
        exp_str = '_'.join([dataset, ] + [f'_[{mode}]'])
        path = {
            'log_dir': './runs',
            'EndoCV': {'data_dir': '../../data/EndoCV2020'},
        }


@ex.config_hook
def add_observer(config, command_name, logger):
    if config['isdebug']:
        exp_name = f'debug_{ex.path}_{config["model"]}_{config["exp_str"]}'
    else:
        exp_name = f'{ex.path}_{config["model"]}_{config["exp_str"]}'
    observer = FileStorageObserver(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config
