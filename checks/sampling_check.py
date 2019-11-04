import os
import sys
import cv2
import torch
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

    coco_parent_dir = os.environ['COCO']

    ply_dir = '../3d_data/trial_old'
    start = '''ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header\n'''

    sampler = torch.nn.functional.grid_sample

    test_loader = SparsePointLoader(train=False, parent_dir=coco_parent_dir)
    data_loader = DataLoader(test_loader, batch_size=opt.batch_size, num_workers=2)

    sampled_image = 0
    selected_image = 0

    for idx, batch in enumerate(data_loader):
        batch_size = batch['num_points'].shape[0]
        im_size = batch['image'].shape[-1] - 1
        # masked_output = output[1] * (batch['mask_image'] > 0).float()
        batch['num_points'] = batch['num_points'].long()
        xy_loc = torch.flip(batch['yx_loc'], dims=(-1,)).long().float()
        # xy_loc = batch['yx_loc']
        xy_loc /= 255
        xy_loc = (xy_loc * 2) - 1
        num = batch['num_points'].view(batch_size, 1)
        for pdx in range(3, batch_size):
            selected_yx = batch['yx_loc'][pdx, :num[pdx, 0], :].long()

            selected_xy = xy_loc[pdx, :num[pdx, 0], :]
            selected_xy = selected_xy.view((1, 1,) + selected_xy.shape)
            noc_gt = batch['noc_points'][pdx, :num[pdx, 0], :]
            noc_gt = noc_gt.cpu().numpy()
            sampled_image = sampler(input=batch['image'][pdx: pdx + 1], grid=selected_xy,
                                    mode='nearest', padding_mode='border')
            sampled_image = sampled_image.view(3, num[pdx, 0])
            sampled_image = sampled_image.cpu().numpy().T * 255
            sampled_image = sampled_image[:, [2, 1, 0]]
            selected_image = batch['image'][pdx, :, batch['yx_loc'][pdx, :num[pdx, 0], 0], batch['yx_loc'][pdx, :num[pdx, 0], 1]]
            # selected_image = selected_image.view(num[pdx, 0], 3)
            selected_image = selected_image.cpu().numpy().T
            selected_image = selected_image[:, [2, 1, 0]]  * 255
            sampled_gt = np.concatenate((noc_gt, sampled_image), axis=1)
            selected_gt = np.concatenate((noc_gt, selected_image), axis=1)

            image = batch['image'][pdx, :, :, :].cpu().numpy()
            image = image.transpose(1, 2, 0) * 255
            cv2.imwrite(ply_dir + '_image.png', image.astype('uint8'))
            start = start.format(num[pdx, 0].item())
            with open(ply_dir + '_sampled_noc.ply', 'w') as write_file:
                write_file.write(start)
                np.savetxt(write_file, sampled_gt, fmt=' '.join(['%0.12f'] * 3 + ['%d'] * 3))

            with open(ply_dir + '_selected_noc.ply', 'w') as write_file:
                write_file.write(start)
                np.savetxt(write_file, selected_gt, fmt=' '.join(['%0.12f'] * 3 + ['%d'] * 3))

            break
        break





    return True


if __name__ == "__main__":
    opt = parse_args(['--log_dir=../data/logs/sparse_sampled_test_ResUNet_Dropout_2Head_3',
                      '--log_iter=200',
                      '--batch_size=8',
                      '--checkpoint=../saves/sparse_sampled_train_ResUNet_Dropout_2Head_2/save_33049.pth',
                      '--num_heads=two',
                      '--model_type=res_unet'] + sys.argv[1:])
    main(opt=opt)