import argparse
import traceback
import time
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import config as cfg
from data_runner import *
from scorenet_runner import *
from scorenet2_runner import *
from initial_runner import *
from train_runner import *

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--testdataset', type=str, default=cfg.test_dataset)
    parser.add_argument('--victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"',
                        default=cfg.VICTIM_DIR)
    parser.add_argument('--attack_model_name', metavar='MODEL_ARCH', type=str, help='Model name',
                        default=cfg.attack_model_arch)
    parser.add_argument("--initial_seed", default=cfg.initial_seed, help="intial seed")
    parser.add_argument('--attack_model_dir', metavar='DIR', type=str, help='Directory containing transferset.pickle',
                        default=cfg.attack_model_dir)
    parser.add_argument('-e', '--epochs', type=int, default=cfg.epoch, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('-b', '--batch_size', metavar='TYPE', type=int, help='Batch size of queries',
                        default=cfg.batch_size)
    parser.add_argumetn('--data_type', type=str, default='GAUSSIAN')
    parser.add_argument('--N_query', type=int, help='quereid data size', default=20000)
    parser.add_argument('--N_test', type=int, help='test data size', default=500)
    parser.add_argument('--channel', type=int, help='data channel', default=1)
    parser.add_argument('--width', type=int, help='data width/length', default=5)

    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    return args

def main():
    args = parse_args_and_config()
    print(">" * 80)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    runner_list = ['DataRunner', 'ScorenetRunner', 'IniitalRunner'] #'FitGuassian' 'TrainRunner'
    for runner in runner_list:
        try:
            runner = eval(runner)(args, device)
            runner.run()
        except:
            logging.error(traceback.format_exc())

    return 0

if __name__ == '__main__':
    sys.exit(main())