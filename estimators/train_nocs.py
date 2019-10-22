"""
Train NOC class
"""
import os
from recordclass import recordclass
from torchstat import stat

from models.networks import *
from utils.common import *


def visualize(batch, output, writer, name, niter):
    writer.add_scalar('L1-Loss', output[1].total_loss, niter)
    writer.add_scalar('NOC-Loss', output[1].NOC_loss, niter)
    writer.add_scalar('Background-Loss', output[1].background_loss, niter)
    write_image(writer, name="{}/Output_NOC".format(name), sample=(output[0][1] * 2) - 1, niter=niter)
    write_image(writer, name="{}/Output_Foreground".format(name),
                sample=((torch.softmax(output[0][0], 1)[:, 1:2, :, :] > 0.5).long() * 2) - 1, niter=niter)
    final_noc = output[0][1] * (torch.softmax(output[0][0], 1)[:, 1:2, :, :] > 0.5).float()
    write_image(writer, name="{}/Output_Final_NOC".format(name), sample=(final_noc * 2) - 1, niter=niter)
    write_image(writer, name="{}/Input".format(name), sample=batch['image'], niter=niter)
    write_image(writer, name="{}/Ground Truth NOC".format(name), sample=batch['noc_image'], niter=niter)
    write_image(writer, name="{}/Ground Truth Foreground".format(name),
                sample=((1 - batch['background']).long() * 2 - 1), niter=niter)
    return True


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class TrainNOCs:
    def __init__(self, save_dir='Trial', mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), num_downs=5, lr=5e-4,
                 betas=(0.5, 0.999), checkpoint=None):

        # self.seg_net = UnetGenerator(input_nc=3, output_nc=3, num_downs=num_downs,
        #                              use_dropout=False, norm_layer=torch.nn.BatchNorm2d,
        #                              last_layer=nn.LeakyReLU(0.2))
        # self.seg_net = Unet2HeadGenerator(input_nc=3, output_nc=3, num_downs=num_downs,
        #                                   use_dropout=False, norm_layer=torch.nn.BatchNorm2d,
        #                                   last_layer=nn.ReLU())
        # self.seg_net = ResNetGenerator(out_channels=3, last_layer=nn.ReLU())
        self.seg_net = ResNet2HeadGenerator(out_channels=3, last_layer=nn.ReLU())
        # self.seg_net = ResUnetGenerator(output_nc=3)

        # self.seg_net.apply(init_weights)
        # stat(model=self.seg_net, input_size=(3, 256, 256))

        self.sparse_l1 = torch.nn.SmoothL1Loss(reduction='none')
        self.l1 = torch.nn.SmoothL1Loss()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.seg_net.cuda()

        self.sparse_l1.cuda()
        self.l1.cuda()
        self.bce.cuda()
        self.seg_net.train()

        self.criterion_selection = None

        self.lr = lr
        self.optimizer = torch.optim.Adam(params=self.seg_net.parameters(), lr=lr)
        # self.optimizer = torch.optim.SGD(params=self.seg_net.parameters(), lr=lr, momentum=0.5)
        if checkpoint is not None:
            checkpoint = torch.load(checkpoint)
            self.seg_net.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.save_path = os.path.join("../saves", save_dir)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.forward = self.forward_2_heads

        self.mean = mean
        self.std = std

        self.loss_tuple = recordclass('losses', ('total_loss', 'NOC_loss', 'background_loss'))

        self.un_norm = UnNormalize(mean=self.mean, std=self.std)

    def criterion_2_heads(self, output, batch):
        return True

    def criterion_l1_sparse(self, output, batch):
        batch_size = batch['num_points'].shape[0]
        # num_points[num_points == 0] = 1
        # sub = torch.abs(output - target)
        # sub = self.sparse_l1(output, target)
        # sub_batch = torch.sum(sub, dim=(1, 2, 3))
        # sub_batch = sub_batch / (num_points * 3)
        batch['num_points'] = batch['num_points'].long()
        num = batch['num_points'].view(batch_size, 1)
        batch['yx_loc'] = batch['yx_loc'].long()
        sub = 0
        for idx in range(batch_size):
            if num[idx, 0] == 0:
                batch_size -= 1
                continue
            sub += self.l1(output[idx, :, batch['yx_loc'][idx, :num[idx, 0], 0], batch['yx_loc'][idx, :num[idx, 0], 1]],
                          batch['noc_image'][idx, :, batch['yx_loc'][idx, :num[idx, 0], 0], batch['yx_loc'][idx, :num[idx, 0], 1]])

        return sub / batch_size

    def forward_sparse(self, batch):
        total_loss = 0
        output = self.seg_net(batch['image'])

        # if 'mask_image' in batch.keys():
        masked_output = output * (batch['mask_image'] > 0).float()
        noc_loss = self.criterion_l1_sparse(output=masked_output, batch=batch)
        total_loss += noc_loss
        # loss = self.l1(masked_output, batch['noc_image'])
        background_target = torch.zeros_like(output)
        background_loss = self.l1(output * batch['background'], background_target)
        total_loss += background_loss
        losses = self.loss_tuple(total_loss=total_loss, NOC_loss=noc_loss, background_loss=background_loss)
        return output, losses

    def forward_2_heads(self, batch):
        total_loss = 0
        output = self.seg_net(batch['image'])
        masked_output = output[1] * (batch['mask_image'] > 0).float()
        noc_loss = self.criterion_l1_sparse(output=masked_output, batch=batch) * 5
        total_loss += noc_loss
        # print(torch.max(((1 - batch['background'][:, 0:1, :, :]) > 0).float()))
        foreground = (1 - batch['background'][:, 0:2, :, :]).float()
        foreground[:, 0, :, :] = batch['background'][:, 0, :, :]
        background_loss = self.bce(input=output[0], target=foreground)
        total_loss += background_loss
        losses = self.loss_tuple(total_loss=total_loss, NOC_loss=noc_loss, background_loss=background_loss)
        return output, losses

    def train(self, batch):
        output, losses = self.forward(batch=batch)
        # output = self.seg_net(batch['image'])
        #
        # # if 'mask_image' in batch.keys():
        # masked_output = output * batch['mask_image']
        # # loss = self.criterion_l1(output=masked_output, batch=batch)
        # l1_loss = self.loss_l1(masked_output, batch['noc_image']) / (
        #             torch.sum(batch['num_points']) + 1 * (torch.sum(batch['num_points'] == 0).float()))
        #
        self.optimizer.zero_grad()
        losses.total_loss.backward()
        self.optimizer.step()

        return output, losses

    def test(self, test_loader, niter, test_writer):
        total_losses = self.loss_tuple(0, 0, 0)

        with torch.no_grad():
            self.seg_net.eval()
            for idx, batch in enumerate(test_loader):
                for keys in batch:
                    batch[keys] = batch[keys].float().cuda()
                # batch['image'] = batch['image'] * 2 - 1
                output, losses = self.forward(batch=batch)

                # masked_output = output[1] * (batch['mask_image'] > 0).float()

                for jdx, val in enumerate(losses):
                    total_losses[jdx] += losses[jdx].item()
                # total_losses.total_loss += losses.total_loss.item()

                if idx == (len(test_loader) - 1):
                    # print(len(test_loader))
                    for jdx, val in enumerate(total_losses):
                        total_losses[jdx] /= len(test_loader)
                    print("Validation loss: {}".format(total_losses.total_loss))

                    # batch['image'] = self.un_norm(batch['image'])
                    batch['image'] = batch['image'] * 2 - 1

                    visualize(writer=test_writer, batch=batch, output=(output, total_losses),
                              name="Validation", niter=niter)

    def run(self, opt, data_loader, writer, epoch=0):

        total_losses = self.loss_tuple(0, 0, 0)
        data_length = len(data_loader.train)

        self.seg_net.train()
        if epoch > 0 and epoch % 15 == 0:
            self.lr *= 0.9
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

        for idx, batch in enumerate(data_loader.train):
            for keys in batch:
                batch[keys] = batch[keys].float().cuda()
            # batch['image'] = batch['image'] * 2 - 1
            output, losses = self.train(batch=batch)

            # print(batch['num_points'], torch.sum(batch['num_points']))
            for jdx, val in enumerate(losses):
                total_losses[jdx] += losses[jdx].item()
            # total_loss += loss.item()
            niter = idx + (epoch * data_length)
            if idx % opt.log_iter == 0:

                print("Epoch: {}  |  Iteration: {}  |  Train Loss: {}".format(epoch, niter, losses.total_loss.item()))
                batch['image'] = batch['image'] * 2 - 1
                visualize(writer=writer.train, batch=batch, output=(output, losses),
                          name="Train Total", niter=niter)

                # batch['image'] = self.un_norm(batch['image'])


                # visualize(writer=writer.train, batch=batch, output=(output, loss),
                #           name="Train", niter=niter)
            if idx == (data_length - 1):
                for jdx, val in enumerate(total_losses):
                    total_losses[jdx] /= data_length
                print("Epoch: {}  |  Final Iteration: {}  |  Train Loss: {}".format(epoch, niter,
                                                                                    total_losses.total_loss))

                batch['image'] = batch['image'] * 2 - 1

                torch.save({'epoch': epoch,
                            'model': self.seg_net.state_dict(),
                            'optimizer': self.optimizer.state_dict()},
                           os.path.join(self.save_path, 'save_{}.pth'.format(niter)))
                # visualize(writer=writer.train, batch=batch, output=(output, total_losses),
                #           name="Train Total", niter=(epoch + 1) * data_length)

                # self.test(test_loader=data_loader.test, test_writer=writer.test, niter=(epoch + 1) * data_length)
        self.test(test_loader=data_loader.test, test_writer=writer.test, niter=(epoch + 1) * data_length + 1)

        print("*" * 100)


if __name__ == "__main__":
    noc_class = TrainNOCs()
