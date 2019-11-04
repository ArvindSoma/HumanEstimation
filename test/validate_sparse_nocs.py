import os
import sys
import numpy as np
import argparse
from tensorboardX import SummaryWriter
from collections import namedtuple

from estimators.train_nocs import TrainNOCs
from data_loaders.densepose_data_loader import SparsePointLoader, DataLoader
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
    parser.add_argument('--log_iter', type=int, default=100, help='logging iteration')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--checkpoint', type=str, default='../saves/', help='checkpoint file')
    parser.add_argument('--model_type', type=str, default='res', help='model type')
    parser.add_argument('--num_heads', type=str, default='one', help='number of output heads')
    return parser.parse_args(args)


def main(opt):

    noc_trained = TrainNOCs(batch_size=opt.batch_size, save_dir=os.path.basename(opt.log_dir),
                            checkpoint=opt.checkpoint, output_heads=opt.num_heads,
                            model_type=opt.model_type)

    coco_parent_dir = os.environ['COCO']

    main_writer = SummaryWriter(os.path.join(opt.log_dir, 'test'))
    test_loader = SparsePointLoader(train=False, parent_dir=coco_parent_dir)
    data_loader = DataLoader(test_loader, batch_size=opt.batch_size, num_workers=2)

    ply_save = '../3d_data/{}'.format(os.path.basename(opt.log_dir))

    noc_trained.validate(test_loader=data_loader, test_writer=main_writer, niter=0, write_ply=True, ply_dir=ply_save)

    return True


if __name__ == "__main__":
    opt = parse_args(['--log_dir=../data/logs/sparse_sampled_test_ResUNet_Dropout_2Head_3',
                      '--log_iter=200',
                      '--batch_size=8',
                      '--checkpoint=../saves/sparse_sampled_train_ResUNet_Dropout_2Head_2/save_33049.pth',
                      '--num_heads=two',
                      '--model_type=res_unet'] + sys.argv[1:])
    main(opt=opt)

