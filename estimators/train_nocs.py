"""
Train NOC class
"""

import torch
import tensorboardX

from models.networks import UnetGenerator
from utils.common import *


def visualize(batch, output, writer, name, niter):
    writer.add_scalar('L1-Loss', output[1].item(), niter)
    write_image(writer, name="{}/Output".format(name), sample=output[0], niter=niter)
    write_image(writer, name="{}/Input".format(name), sample=batch['image'], niter=niter)
    write_image(writer, name="{}/Ground Truth".format(name), sample=batch['noc_image'], niter=niter)
    return True


class TrainNOCs:
    def __init__(self, num_downs=5, lr=1e-2, betas=(0.5, 0.999)):

        self.seg_net = UnetGenerator(input_nc=3, output_nc=3, num_downs=5,
                                     use_dropout=False, norm_layer=torch.nn.InstanceNorm2d,
                                     block_type=1)
        self.loss_l1 = torch.nn.SmoothL1Loss(reduction='sum')
        self.seg_net.cuda()
        self.loss_l1.cuda()
        self.seg_net.train()

        self.lr = lr

        self.optimizer = torch.optim.Adam(params=self.seg_net.parameters(), lr=self.lr, betas=betas)
        self.iter = 0

        self.mean = (0, 0, 0)
        self.std = (0.5, 0.5, 0.5)

    def criterion_l1(self, output, batch):
        return self.loss_l1(output, batch['noc_image']) / (torch.sum(batch['num_points']) + 1 * (torch.sum(batch['num_points'] == 0).float()))

    def criterion_l1_sparse(self, output, batch):
        return self.loss_l1(output, batch['noc_image']) / (torch.sum(batch['num_points']) + 1 * (torch.sum(batch['num_points'] == 0).float()))

    def forward(self, batch):
        masked_output = 0
        output = self.seg_net(batch['image'])

        if 'mask_image' in batch.keys():
            masked_output = output * batch['mask_image']
        loss = self.criterion_l1(output=masked_output, batch=batch)

        return output, loss

    def train(self, batch):
        output, l1_loss = self.forward(batch=batch)

        self.optimizer.zero_grad()
        l1_loss.backward()
        self.optimizer.step()

        return output, l1_loss

    def test(self, test_loader, niter, test_writer):
        total_loss = 0
        for idx, batch in enumerate(test_loader):
            for keys in batch:
                batch[keys] = batch[keys].cuda().float()
            batch['image'] = batch['image'] * 2 - 1
            output, l1_loss = self.forward(batch=batch)

            total_loss += l1_loss

            if idx is len(test_loader) - 1:
                final_loss = total_loss / len(test_loader)
                print("Validation loss: {}".format(final_loss))
                visualize(writer=test_writer, batch=batch, output=(output, final_loss),
                          name="Validation", niter=niter)

    def run(self, opt, data_loader, writer, epoch=0):

        data_length = len(data_loader.train)

        if epoch > 0 and epoch % 25:
            self.lr *= 0.9
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

        for idx, batch in enumerate(data_loader.train):
            for keys in batch:
                batch[keys] = batch[keys].float().cuda()
            batch['image'] = batch['image'] * 2 - 1
            output, l1_loss = self.train(batch=batch)

            if idx % opt.log_iter == 0:
                niter = (idx + 1) + (epoch * data_length)
                print("Epoch: {}  |  Iteration: {}  |  Train Loss: {}".format(epoch, niter, l1_loss.item()))
                visualize(writer=writer.train, batch=batch, output=(output, l1_loss),
                          name="Train", niter=niter)
        with torch.no_grad():
            self.test(test_loader=data_loader.test, test_writer=writer.test, niter=(epoch + 1) * data_length)

        print("*" * 100)



