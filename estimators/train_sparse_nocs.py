import os
import sys
import numpy as np
import argparse
from collections import namedtuple

from data_loaders.densepose_data_loader import SparsePointLoader
from utils.common import *


def parse_args(args):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=32, help='# of samples of human poses')
    return parser.parse_args(args)


def main(opt):

     return True


if __name__ == "__main__":
     opt = parse_args([
          '--n_samples=10'])

     main(opt=opt)
