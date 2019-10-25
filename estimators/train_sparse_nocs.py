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
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--num_heads', type=str, default='one', help='number of output heads')
    return parser.parse_args(args)


def main(opt):

    noc_trained = TrainNOCs(save_dir=os.path.basename(opt.log_dir), output_heads=opt.num_heads)

    coco_parent_dir = os.environ['COCO']

    main_writer = namedtuple('main_writer', ('train', 'validate'))
    main_writer = main_writer(SummaryWriter(os.path.join(opt.log_dir, 'train')),
                              SummaryWriter(os.path.join(opt.log_dir, 'test')))

    train_loader = SparsePointLoader(train=True, parent_dir=coco_parent_dir)
    train_loader = DataLoader(train_loader, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    test_loader = SparsePointLoader(train=False, parent_dir=coco_parent_dir)
    test_loader = DataLoader(test_loader, batch_size=opt.batch_size, num_workers=2)

    data_loader = namedtuple('data_loader', ('train', 'validate'))
    data_loader = data_loader(train_loader, test_loader)
    for epoch in range(opt.epochs):
        noc_trained.run(opt=opt, data_loader=data_loader, writer=main_writer, epoch=epoch)

    return True


if __name__ == "__main__":
    opt = parse_args(['--log_dir=../data/logs/sparse_trial_ResNet_Dropout_2Head_1',
                      '--log_iter=200',
                      '--batch_size=8',
                      '--epochs=100',
                      '--num_heads=two'] + sys.argv[1:])
    main(opt=opt)

