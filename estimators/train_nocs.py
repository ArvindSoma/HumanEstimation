"""
Train NOC class
"""

import torch
import tensorboardX

from models.networks import UnetGenerator
from utils.common import *


class TrainNOCs:
    def __init__(self, lr=1e-4, betas=(0.5, 0.999)):

        self.seg_net = UnetGenerator(input_nc=3, output_nc=3, num_downs=5,
                                     use_dropout=False, norm_layer=torch.nn.InstanceNorm2d)
        self.loss_l1 = torch.nn.SmoothL1Loss()
        self.setup()

        self.optimizer = torch.optim.Adam(params=self.seg_net.parameters(), lr=lr, betas=betas)
        self.iter = 0

        self.mean = (0, 0, 0)
        self.std = (0.5, 0.5, 0.5)

    def setup(self):
        self.seg_net.cuda()
        self.loss_l1.cuda()
        self.seg_net.train()

    def train(self, batch):
        output = self.seg_net(batch['image'])
        if 'mask_image' in batch.keys():
            output = output * batch['mask_image']

        self.optimizer.zero_grad()
        l1_loss = self.loss_l1(output)
        l1_loss.backward()
        self.optimizer.step()

        return output, l1_loss

    def run(self, opt, batch_loader, writer, epoch=0):
        data_length = len(batch_loader)
        for idx, batch in enumerate(batch_loader):
            for keys in batch:
                batch[keys] = batch[keys].cuda()

            output = self.train(batch=batch)

            if idx % opt.log_iter == 0:
                self.visualize(batch=batch, output=output, niter=(idx + 1) + (epoch * data_length))

    def visualize(self, batch, output, writer, niter):
        writer.train.add_scalar('L1-Loss', output[1].item(), niter)
        output[0] = output[0]
        write_image(writer.train, name="Train", sample=output[0], niter=niter)
        return True


