"""
Train NOC class
"""
import os
import cv2
from recordclass import recordclass
from torchstat import stat
from math import log10
from models.networks import *
from utils.common import *


def visualize(batch, output, writer, name, niter, foreground=False, test=False, viz_loss=True, part_segmentation=False):
    output_image = output[0]
    if foreground:
        output_image = output[0][1]
        foreground = output[0][0]
        write_image(writer, name="{}/Ground Truth Foreground".format(name),
                    sample=((1 - batch['background']).long() * 2 - 1), niter=niter)
        write_image(writer, name="{}/Output_Foreground".format(name),
                    sample=((torch.softmax(foreground, 1)[:, 1:2, :, :] > 0.5).long() * 2) - 1, niter=niter)
        final_noc = output_image * (torch.softmax(output[0][0], 1)[:, 1:2, :, :] > 0.5).float()
        write_image(writer, name="{}/Output_Final_NOC".format(name), sample=(final_noc * 2) - 1, niter=niter)

    if part_segmentation:
        writer.add_scalar('Part_Segmentation-Loss', output[1].part_segmentation_loss, niter)
    write_image(writer, name="{}/Output_NOC".format(name), sample=(output_image * 2) - 1, niter=niter)
    write_image(writer, name="{}/Input".format(name), sample=batch['image'], niter=niter)
    if viz_loss:
        writer.add_scalar('L1-Loss', output[1].total_loss, niter)
        writer.add_scalar('NOC-Loss', output[1].NOC_loss, niter)
        writer.add_scalar('Background-Loss', output[1].background_loss, niter)
    if not test:
        ground_image = torch.zeros_like(batch['image'])
    #     write_image(writer, name="{}/Ground Truth NOC".format(name), sample=batch['noc_image'], niter=niter)

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
    def __init__(self, save_dir='Trial', mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), lr=2e-4,
                 betas=(0.5, 0.999), batch_size=8, checkpoint=None, model_type='res', output_heads='two',
                 backbone='res18', use_dropout=True):
        if backbone == 'res50':
            latent = ResNet50Features(final_layer=-2)
            ngf = 256
        else:
            latent = ResNet18Features(final_layer=-2)
            ngf = 64
        # index_list = [8, 7, 6, 5, 4, 3]

        self.part_segmentation = False
        self.part_classes = 25
        if output_heads == 'one':
            self.forward = self.forward_sparse
            self.seg_net = ResNetGenerator(latent=latent, out_channels=3,
                                           last_layer=nn.ReLU())
            if model_type == 'res_unet':
                self.seg_net = ResUnetGenerator(latent=latent, output_nc=3,
                                                last_layer=nn.ReLU(), ngf=ngf)

            # self.seg_net = UnetGenerator(input_nc=3, output_nc=3, num_downs=num_downs,
            #                              use_dropout=False, norm_layer=torch.nn.BatchNorm2d,
            #                              last_layer=nn.LeakyReLU(0.2))
            self.foreground = False
        elif output_heads == 'two':
            self.forward = self.forward_2_heads
            self.seg_net = ResNet2HeadGenerator(latent=latent, out_channels=3, last_layer=nn.ReLU())
            if model_type == 'res_unet':
                self.seg_net = ResUnet2HeadGenerator(latent=latent, output_nc=3,
                                                     last_layer=nn.ReLU(), ngf=ngf)
            # self.seg_net = Unet2HeadGenerator(input_nc=3, output_nc=3, num_downs=num_downs,
            #                                   use_dropout=False, norm_layer=torch.nn.BatchNorm2d,
            #                                   last_layer=nn.ReLU())
            self.foreground = True
        elif output_heads == 'three':
            self.forward = self.forward_3_heads
            self.seg_net = ResNet2HeadGenerator(latent=latent, out_channels=3, last_layer=nn.ReLU())
            if model_type == 'res_unet':
                self.seg_net = ResUnet3HeadGenerator(latent=latent, output_nc=3,
                                                     last_layer=nn.ReLU(), ngf=ngf)
            # self.seg_net = Unet2HeadGenerator(input_nc=3, output_nc=3, num_downs=num_downs,
            #                                   use_dropout=False, norm_layer=torch.nn.BatchNorm2d,
            #                                   last_layer=nn.ReLU())
            self.part_segmentation = True
            self.foreground = True
        else:
            self.foreground = None
            print("Error! Unknown number of heads!")
            exit(256)

        print("Using model {}.".format(self.seg_net.__class__.__name__))

        # self.seg_net.apply(init_weights)
        # stat(model=self.seg_net, input_size=(3, 256, 256))
        self.sampler = torch.nn.functional.grid_sample
        self.sparse_l1 = torch.nn.SmoothL1Loss(reduction='none')
        self.l1 = torch.nn.SmoothL1Loss()
        self.l2 = torch.nn.MSELoss()
        self.distance = torch.nn.PairwiseDistance(p=2)
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.ce = torch.nn.CrossEntropyLoss()
        self.seg_net.cuda()

        self.sparse_l1.cuda()
        self.l1.cuda()
        self.bce.cuda()
        self.distance.cuda()
        self.seg_net.train()

        self.criterion_noc = self.criterion_l1_sparse_bilinear

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

        self.batch_size = batch_size
        self.mean = mean
        self.std = std

        if not self.part_segmentation:
            self.loss_tuple = recordclass('losses',
                                          ('total_loss', 'NOC_loss', 'background_loss', 'NOC_mse', 'NOC_distance'))
        else:
            self.loss_tuple = recordclass('losses',
                                          ('total_loss', 'NOC_loss', 'background_loss', 'NOC_mse', 'NOC_distance',
                                           'part_segmentation_loss'))

        self.ply_start = '''ply
        format ascii 1.0
        element vertex {}
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header\n'''

        self.un_norm = UnNormalize(mean=self.mean, std=self.std)

    def get_loss_tuple(self):
        if self.part_segmentation:
            return self.loss_tuple(0, 0, 0, 0, 0, 0)
        else:
            return self.loss_tuple(0, 0, 0, 0, 0)

    def criterion_distance(self, output, batch):
        batch_size = batch['num_points'].shape[0]
        im_size = batch['image'].shape[-1]
        batch['num_points'] = batch['num_points'].long()
        num = batch['num_points'].view(batch_size, 1)
        # batch['yx_loc'] = batch['yx_loc'].view((1, 1, ) + batch['yx_loc'].shape)
        xy_loc = torch.flip(batch['yx_loc'], dims=(-1,)).float()
        xy_loc /= 255
        xy_loc = (xy_loc * 2) - 1
        sub = 0
        for idx in range(batch_size):
            if num[idx, 0] == 0:
                # batch_size -= 1
                continue
            selected_xy = xy_loc[idx, :num[idx, 0], :]
            selected_xy = selected_xy.view((1, 1,) + selected_xy.shape)
            selected_noc = batch['noc_points'][idx, :num[idx, 0], :]
            # selected_noc = torch.transpose(torch.transpose(selected_noc, 0, 2), 1, 2)
            # selected_noc = selected_noc.view((1,) + selected_noc.shape)
            sampled_indices = self.sampler(input=output[idx: (idx + 1)], grid=selected_xy,
                                           mode='bilinear', padding_mode='border')
            sampled_indices = sampled_indices.view(3, num[idx, 0])
            sampled_indices = torch.transpose(sampled_indices, 0, 1)
            sub += torch.mean(self.distance(sampled_indices, selected_noc))

        return sub / batch_size

    def criterion_mse(self, output, batch):
        # batch_size = batch['num_points'].shape[0]
        #
        # batch['num_points'] = batch['num_points'].long()
        # num = batch['num_points'].view(batch_size, 1)
        # batch['yx_loc'] = batch['yx_loc'].long()
        # sub = 0
        # for idx in range(batch_size):
        #     if num[idx, 0] == 0:
        #         batch_size -= 1
        #         continue
        #     sub += self.l2(output[idx, :, batch['yx_loc'][idx, :num[idx, 0], 0], batch['yx_loc'][idx, :num[idx, 0], 1]],
        #                    batch['noc_image'][idx, :, batch['yx_loc'][idx, :num[idx, 0], 0],
        #                    batch['yx_loc'][idx, :num[idx, 0], 1]])
        batch_size = batch['num_points'].shape[0]
        im_size = batch['image'].shape[-1] - 1
        batch['num_points'] = batch['num_points'].long()
        num = batch['num_points'].view(batch_size, 1)
        # batch['yx_loc'] = batch['yx_loc'].view((1, 1, ) + batch['yx_loc'].shape)
        xy_loc = torch.flip(batch['yx_loc'], dims=(-1,)).float()
        xy_loc /= 255
        xy_loc = (xy_loc * 2) - 1
        sub = 0
        for idx in range(batch_size):
            if num[idx, 0] == 0:
                # batch_size -= 1
                continue
            selected_xy = xy_loc[idx, :num[idx, 0], :]
            selected_xy = selected_xy.view((1, 1,) + selected_xy.shape)
            selected_noc = batch['noc_points'][idx, :num[idx, 0], :]
            # selected_noc = torch.transpose(torch.transpose(selected_noc, 0, 2), 1, 2)
            # selected_noc = selected_noc.view((1,) + selected_noc.shape)
            sampled_indices = self.sampler(input=output[idx: (idx + 1)], grid=selected_xy,
                                           mode='bilinear', padding_mode='border')
            sampled_indices = sampled_indices.view(3, num[idx, 0])
            sampled_indices = torch.transpose(sampled_indices, 0, 1)
            sub += self.l2(sampled_indices, selected_noc)

        return sub / batch_size

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
                # batch_size -= 1
                continue
            sub += self.l1(output[idx, :, batch['yx_loc'][idx, :num[idx, 0], 0], batch['yx_loc'][idx, :num[idx, 0], 1]],
                           batch['noc_image'][idx, :, batch['yx_loc'][idx, :num[idx, 0], 0],
                           batch['yx_loc'][idx, :num[idx, 0], 1]])

        return sub / batch_size

    def criterion_l1_sparse_bilinear(self, output, batch):
        batch_size = batch['num_points'].shape[0]
        im_size = batch['image'].shape[-1]
        batch['num_points'] = batch['num_points'].long()
        num = batch['num_points'].view(batch_size, 1)
        # batch['yx_loc'] = batch['yx_loc'].view((1, 1, ) + batch['yx_loc'].shape)
        xy_loc = torch.flip(batch['yx_loc'], dims=(-1,)).float()
        xy_loc /= 255
        xy_loc = (xy_loc * 2) - 1
        sub = 0
        for idx in range(batch_size):
            if num[idx, 0] == 0:
                # batch_size -= 1
                continue
            selected_xy = xy_loc[idx, :num[idx, 0], :]
            selected_xy = selected_xy.view((1, 1,) + selected_xy.shape)
            selected_noc = batch['noc_points'][idx, :num[idx, 0], :]
            # selected_noc = torch.transpose(torch.transpose(selected_noc, 0, 2), 1, 2)
            # selected_noc = selected_noc.view((1,) + selected_noc.shape)
            sampled_indices = self.sampler(input=output[idx: (idx + 1)], grid=selected_xy,
                                           mode='bilinear', padding_mode='border')
            sampled_indices = sampled_indices.view(3, num[idx, 0])
            sampled_indices = torch.transpose(sampled_indices, 0, 1)
            sub += self.l1(sampled_indices, selected_noc)

        return sub / batch_size

    def criterion_ce_sparse_bilinear(self, output, batch):
        batch_size = batch['num_points'].shape[0]
        im_size = batch['image'].shape[-1]
        batch['num_points'] = batch['num_points'].long()
        num = batch['num_points'].view(batch_size, 1)
        # batch['yx_loc'] = batch['yx_loc'].view((1, 1, ) + batch['yx_loc'].shape)
        xy_loc = torch.flip(batch['yx_loc'], dims=(-1,)).float()
        xy_loc /= 255
        xy_loc = (xy_loc * 2) - 1
        sub = 0
        for idx in range(batch_size):
            if num[idx, 0] == 0:
                # batch_size -= 1
                continue
            selected_xy = xy_loc[idx, :num[idx, 0], :]
            selected_xy = selected_xy.view((1, 1,) + selected_xy.shape)
            selected_classes = batch['patch_points'][idx, :num[idx, 0], 0].long()
            # selected_noc = torch.transpose(torch.transpose(selected_noc, 0, 2), 1, 2)
            # selected_noc = selected_noc.view((1,) + selected_noc.shape)
            sampled_indices = self.sampler(input=output[idx: (idx + 1)], grid=selected_xy,
                                           mode='bilinear', padding_mode='border')
            sampled_indices = sampled_indices.view(self.part_classes, num[idx, 0])
            sampled_indices = torch.transpose(sampled_indices, 0, 1)
            sub += self.ce(sampled_indices, selected_classes)

        return sub / batch_size

    def forward_sparse(self, batch):
        total_loss = 0
        output = self.seg_net(batch['image'])

        # if 'mask_image' in batch.keys():
        # masked_output = output * (batch['mask_image'] > 0).float()
        noc_loss = self.criterion_noc(output=output, batch=batch)
        total_loss += noc_loss * 10
        # loss = self.l1(masked_output, batch['noc_image'])
        background_target = torch.zeros_like(output)
        background_loss = self.l1(output * batch['background'], background_target)
        total_loss += background_loss
        mse = self.criterion_mse(output=output, batch=batch)
        distance = self.criterion_distance(output=output, batch=batch)
        losses = self.loss_tuple(total_loss=total_loss, NOC_loss=noc_loss,
                                 background_loss=background_loss, NOC_mse=mse, NOC_distance=distance)
        return output, losses

    def forward_2_heads(self, batch):
        total_loss = 0
        output = self.seg_net(batch['image'])
        # masked_output = output[1] * (batch['mask_image'] > 0).float()
        noc_loss = self.criterion_noc(output=output[1], batch=batch)
        total_loss += noc_loss * 100
        # print(torch.max(((1 - batch['background'][:, 0:1, :, :]) > 0).float()))
        foreground = (1 - batch['background'][:, 0:2, :, :]).float()
        foreground[:, 0, :, :] = batch['background'][:, 0, :, :]
        background_loss = self.bce(input=output[0], target=foreground)
        total_loss += background_loss
        # with torch.no_grad():
        distance = self.criterion_distance(output=output[1], batch=batch)
        mse = self.criterion_mse(output=output[1], batch=batch)
        losses = self.loss_tuple(total_loss=total_loss, NOC_loss=noc_loss,
                                 background_loss=background_loss, NOC_mse=mse,
                                 NOC_distance=distance)
        return output, losses

    def forward_3_heads(self, batch):
        total_loss = 0
        output = self.seg_net(batch['image'])
        # masked_output = output[1] * (batch['mask_image'] > 0).float()
        noc_loss = self.criterion_noc(output=output[1], batch=batch)
        total_loss += noc_loss
        # print(torch.max(((1 - batch['background'][:, 0:1, :, :]) > 0).float()))
        foreground = (1 - batch['background'][:, 0:2, :, :]).float()
        foreground[:, 0, :, :] = batch['background'][:, 0, :, :]
        background_loss = self.bce(input=output[0], target=foreground)
        total_loss += background_loss
        patch_loss = self.criterion_ce_sparse_bilinear(output=output[2], batch=batch)
        total_loss += patch_loss
        # with torch.no_grad():
        distance = self.criterion_distance(output=output[1], batch=batch)
        mse = self.criterion_mse(output=output[1], batch=batch)
        losses = self.loss_tuple(total_loss=total_loss, NOC_loss=noc_loss,
                                 background_loss=background_loss, NOC_mse=mse,
                                 NOC_distance=distance,
                                 part_segmentation_loss=patch_loss)
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

    def write_noc_ply(self, output, batch, idx, ply_dir):

        batch_size = batch['num_points'].shape[0]
        im_size = batch['image'].shape[-1] - 1
        idx = idx * self.batch_size
        # masked_output = output[1] * (batch['mask_image'] > 0).float()
        batch['num_points'] = batch['num_points'].long()
        xy_loc = torch.flip(batch['yx_loc'], dims=(-1,)).float()
        # xy_loc = batch['yx_loc']
        xy_loc /= 255
        xy_loc = (xy_loc * 2) - 1
        num = batch['num_points'].view(batch_size, 1)
        for pdx in range(batch_size):
            if self.foreground:
                out_arr = output[1]
            else:
                out_arr = output
            selected_yx = batch['yx_loc'][pdx, :num[pdx, 0], :].long()

            selected_xy = xy_loc[pdx, :num[pdx, 0], :]
            selected_xy = selected_xy.view((1, 1,) + selected_xy.shape)
            # selected_noc = batch['noc_points'][pdx: pdx + 1, :num[pdx, 0]]
            # selected_noc = torch.transpose(torch.transpose(selected_noc, 0, 2), 1, 2)
            # selected_noc = selected_noc.view((1,) + selected_noc.shape)
            sampled_image = self.sampler(input=batch['image'][pdx: pdx + 1], grid=selected_xy,
                                         mode='bilinear', padding_mode='border')
            sampled_image = sampled_image.view(3, num[pdx, 0])
            sampled_image = sampled_image.cpu().numpy().T * 255
            sampled_image = sampled_image[:, [2, 1, 0]]
            output_arr = self.sampler(input=out_arr[pdx: pdx + 1], grid=selected_xy,
                                      mode='bilinear', padding_mode='border')
            # image = batch['image'][pdx, :, selected_yx[:, 0], selected_yx[:, 1]]
            # output_arr = out_arr[pdx, :, batch['yx_loc'][pdx, :num[pdx, 0], 0], batch['yx_loc'][pdx, :num[pdx, 0], 1]]
            noc_gt = batch['noc_points'][pdx, :num[pdx, 0], :].cpu().numpy()
            # image = image.view(3, num[pdx, 0])
            # image = image.cpu().numpy().T * 255
            # image = image[:, [2, 1, 0]]
            output_arr = output_arr.view(3, num[pdx, 0])
            output_arr = output_arr.cpu().numpy().T

            # image = image.astype('uint8')
            # output_arr = output_arr.astype('uint8')

            # image = image[:, [2, 1, 0]] * 255
            # output_arr = output_arr.cpu().numpy().T

            start = self.ply_start.format(num[pdx, 0].item())
            concatenated_out = np.concatenate((output_arr, sampled_image), axis=1)
            concatenated_gt = np.concatenate((noc_gt, sampled_image), axis=1)
            gt_image = batch['image'][pdx, :, :, :].cpu().numpy()
            gt_image = gt_image.transpose(1, 2, 0) * 255
            cv2.imwrite(os.path.join(ply_dir, 'Ground_truth_{}.png'.format(idx + pdx)), gt_image.astype('uint8'))
            with open(os.path.join(ply_dir, 'Output_{}.ply'.format(idx + pdx)), 'w') as write_file:
                write_file.write(start)
                np.savetxt(write_file, concatenated_out, fmt=' '.join(['%0.12f'] * 3 + ['%d'] * 3))

            with open(os.path.join(ply_dir, 'Ground_truth_{}.ply'.format(idx + pdx)), 'w') as write_file:
                write_file.write(start)
                np.savetxt(write_file, concatenated_gt, fmt=' '.join(['%0.12f'] * 3 + ['%d'] * 3))

    def validate(self, test_loader, niter, test_writer, write_ply=False, ply_dir=''):
        total_losses = self.get_loss_tuple()
        mse_metric = 0
        if write_ply:
            if not os.path.exists(ply_dir):
                os.mkdir(ply_dir)
        with torch.no_grad():
            self.seg_net.eval()
            for idx, batch in enumerate(test_loader):
                for keys in batch:
                    batch[keys] = batch[keys].float().cuda()

                output, losses = self.forward(batch=batch)

                #  View NOC as PLY
                if write_ply:
                    self.write_noc_ply(output=output, batch=batch, idx=idx, ply_dir=ply_dir)
                    batch['image'] = batch['image'] * 2 - 1
                    visualize(writer=test_writer, batch=batch, output=(output, total_losses),
                              name="Validation", niter=idx, foreground=self.foreground, viz_loss=False,
                              part_segmentation=self.part_segmentation)

                for jdx, val in enumerate(losses._asdict()):
                    if val == 'NOC_mse':
                        total_losses[jdx] += 10 * log10(1 / losses[jdx].item())
                        mse_metric += losses[jdx].item()
                    else:
                        total_losses[jdx] += losses[jdx].item()
                # total_losses.total_loss += losses.total_loss.item()

                if idx == (len(test_loader) - 1):
                    # print(len(test_loader))
                    mse_metric /= len(test_loader)
                    for jdx, val in enumerate(total_losses):

                        total_losses[jdx] /= len(test_loader)

                    print("Validation loss: {}".format(total_losses.total_loss))

                    # batch['image'] = self.un_norm(batch['image'])

                    test_writer.add_scalar('PSNR', total_losses.NOC_mse, niter)
                    test_writer.add_scalar('MSE', mse_metric, niter)
                    test_writer.add_scalar('Distance', total_losses.NOC_distance, niter)
                    batch['image'] = batch['image'] * 2 - 1
                    visualize(writer=test_writer, batch=batch, output=(output, total_losses),
                              name="Validation", niter=niter, foreground=self.foreground,
                              part_segmentation=self.part_segmentation)

    def test(self, test_loader, niter, test_writer):

        with torch.no_grad():
            self.seg_net.eval()
            for idx, batch in enumerate(test_loader):
                for keys in batch:
                    batch[keys] = batch[keys].float().cuda()
                output = self.seg_net(batch['image'])
                batch['image'] = batch['image'] * 2 - 1
                visualize(writer=test_writer, batch=batch, output=(output, self.loss_tuple(0, 0, 0, 0)),
                          name="Validation", niter=idx, foreground=self.foreground,
                          part_segmentation=self.part_segmentation, test=True)

    def run(self, opt, data_loader, writer, epoch=0):

        total_losses = self.get_loss_tuple()
        mse_metric = 0

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
            for jdx, val in enumerate(losses._asdict()):
                if val == 'NOC_mse':
                    total_losses[jdx] += 10 * log10(1 / losses[jdx].item())
                    mse_metric += losses[jdx].item()
                else:
                    total_losses[jdx] += losses[jdx].item()
            # total_loss += loss.item()
            niter = idx + (epoch * data_length)
            if idx % opt.log_iter == 0:

                print("Epoch: {}  |  Iteration: {}  |  Train Loss: {}".format(epoch, niter, losses.total_loss.item()))
                batch['image'] = batch['image'] * 2 - 1
                visualize(writer=writer.train, batch=batch, output=(output, losses),
                          name="Train Total", niter=niter, foreground=self.foreground,
                          part_segmentation=self.part_segmentation)

                # batch['image'] = self.un_norm(batch['image'])
                # visualize(writer=writer.train, batch=batch, output=(output, loss),
                #           name="Train", niter=niter)
            # Last Iteration
            if idx == (data_length - 1):
                mse_metric /= data_length
                for jdx, val in enumerate(total_losses):
                    total_losses[jdx] /= data_length
                print("Epoch: {}  |  Final Iteration: {}  |  Train Loss: {}".format(epoch, niter,
                                                                                    total_losses.total_loss))

                batch['image'] = batch['image'] * 2 - 1
                writer.train.add_scalar('PSNR', total_losses.NOC_mse, niter)
                writer.train.add_scalar('MSE', mse_metric, niter)
                writer.train.add_scalar('Distance', total_losses.NOC_distance, niter)
                torch.save({'epoch': epoch,
                            'model': self.seg_net.state_dict(),
                            'optimizer': self.optimizer.state_dict()},
                           os.path.join(self.save_path, 'save_{}.pth'.format(niter)))
                # visualize(writer=writer.train, batch=batch, output=(output, total_losses),
                #           name="Train Total", niter=(epoch + 1) * data_length)

                # self.test(test_loader=data_loader.test, test_writer=writer.test, niter=(epoch + 1) * data_length)
        self.validate(test_loader=data_loader.validate, test_writer=writer.validate, niter=(epoch + 1) * data_length + 1)

        print("*" * 100)


if __name__ == "__main__":
    noc_class = TrainNOCs()
