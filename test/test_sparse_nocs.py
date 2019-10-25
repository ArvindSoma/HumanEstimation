import os
import sys
import numpy as np
import argparse
from tensorboardX import SummaryWriter
from collections import namedtuple

from estimators.train_nocs import TrainNOCs
from data_loaders.image_loader import ImageLoader, DataLoader
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
    parser.add_argument('--log_dir', type=str, default='../data/logs', help='log directory')
    parser.add_argument('--parent_dir', type=str, default='../data/logs', help='image parent directory')
    parser.add_argument('--log_iter', type=int, default=100, help='logging iteration')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--checkpoint', type=str, default='../saves/', help='checkpoint file')
    parser.add_argument('--model_type', type=str, default='res', help='model type')
    parser.add_argument('--num_heads', type=str, default='one', help='number of output heads')
    parser.add_argument('--file_ext', type=str, default='png', help='file extension')
    return parser.parse_args(args)


def main(opt):

    noc_trained = TrainNOCs(batch_size=opt.batch_size, save_dir=os.path.basename(opt.log_dir),
                            checkpoint=opt.checkpoint, output_heads=opt.num_heads,
                            model_type=opt.model_type)

    main_writer = SummaryWriter(os.path.join(opt.log_dir, 'test'))
    test_loader = ImageLoader(train=False, parent_dir=opt.parent_dir, file_ext=opt.file_ext)
    data_loader = DataLoader(test_loader, batch_size=opt.batch_size, num_workers=2)

    noc_trained.test(test_loader=data_loader, test_writer=main_writer, niter=0)

    return True


if __name__ == "__main__":
    opt = parse_args(['--log_dir=../data/logs/sparse_test_ResNet_Dropout_2Heads_3',
                      '--log_iter=200',
                      '--batch_size=8',
                      '--checkpoint=../saves/sparse_trial_ResNet_Dropout_2Heads_3/save_82624.pth',
                      '--parent_dir=../3d_data/DensePoseData/demo_data',
                      '--model_type=res_unet',
                      '--num_heads=two',
                      '--file_ext=png'] + sys.argv[1:])
    main(opt=opt)

