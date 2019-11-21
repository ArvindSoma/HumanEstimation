import os
import sys
import numpy as np
import argparse
from tensorboardX import SummaryWriter
from collections import namedtuple

from estimators.train_nocs import TrainNOCs
from data_loaders.densepose_results_loader import DenseDataLoader, DataLoader
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
                            output_heads=opt.num_heads, model_type=opt.model_type)
    coco_parent_dir = os.environ['COCO']
    test_loader = DenseDataLoader(train=False, parent_dir=coco_parent_dir)
    data_loader = DataLoader(test_loader, batch_size=opt.batch_size, num_workers=2)
    loss = 0
    for idx, batch in enumerate(data_loader):
        for keys in batch:
            batch[keys] = batch[keys].float().cuda()
        loss += noc_trained.criterion_l1_sparse_bilinear(output=batch['dense_noc'], batch=batch)

    print("DensePose Loss: {}".format(loss.item() / len(data_loader)))

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

