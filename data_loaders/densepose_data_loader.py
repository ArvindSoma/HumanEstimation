import os
import numpy as np
import time
import cv2
import torch
import random
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class SparsePointLoader(Dataset):
    def __init__(self,
                 train=True,
                 scales=(1.0,),
                 parent_dir='',
                 base_size=None,
                 crop_size=256,
                 point_select=1,
                 flip=True,
                 transform=transforms.Compose([
                     transforms.ToTensor()])):
        # ,
        #                      transforms.Normalize(mean=(0, 0, 0), std=(0.5, 0.5, 0.5))

        self.parent_dir = parent_dir
        self.base_size = base_size
        self.scales = scales
        self.crop_size = crop_size
        self.flip = flip
        self.point_select = point_select
        self.train = train
        self.transform = transform
        self.max_pad = 1200

        if self.train:
            read_file_name = '../data/dp_annotation/train.pkl'

        else:
            read_file_name = '../data/dp_annotation/minival.pkl'  # 'valminusminival'
            self.point_select = 1

        with open(read_file_name, 'rb') as read_file:
            self.data = pickle.load(read_file)

        self.len = len(self.data)

    def fixed_rescale_crop(self, image, background):
        h, w, c = image.shape
        center = not self.train
        # center = True
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)

        img = cv2.resize(image, (ow, oh), cv2.INTER_CUBIC)
        background = cv2.resize(background, (ow, oh), cv2.INTER_NEAREST)
        h, w, _ = img.shape
        if center:
            x1 = int(round((w - self.crop_size) / 2.))
            y1 = int(round((h - self.crop_size) / 2.))
        else:
            x1 = random.randint(0, w - self.crop_size)
            y1 = random.randint(0, h - self.crop_size)
        img = img[y1: y1 + self.crop_size, x1: x1 + self.crop_size, :]
        background = background[y1: y1 + self.crop_size, x1: x1 + self.crop_size, :]

        return img, background, (oh, ow), (y1, x1)

    def __getitem__(self, index):
        """

        :param index: Index value
        :return: Dictionary of data points
        """

        data_dict = {}

        data_point = self.data[index]
        internal_file_loc = data_point['file_name']

        image = cv2.imread(filename=os.path.join(self.parent_dir, internal_file_loc))
        h, w, c = image.shape
        yx_loc = data_point['points']['yx']
        point_len, _ = yx_loc.shape
        noc_points = data_point['points']['noc']

        background = cv2.imread(filename=os.path.join(self.parent_dir, 'background', internal_file_loc))
        # print(background[:, :, 0] == )


        # np.random.seed(int(time.time()))
        # selection = np.random.choice(point_len, round(point_len * self.point_select), replace=False)
        #
        # yx_loc = yx_loc[selection, :]
        # noc_points = noc_points[selection, :]

        data_dict['image'], background, (h_, w_), (c_h, c_w) = self.fixed_rescale_crop(image=image,
                                                                                       background=background)
        data_dict['background'] = torch.from_numpy((background.transpose([2, 0, 1]) > 0).astype('float32'))

        yx_loc = (yx_loc * [h_, w_] / [h, w])

        loc_selection = np.where(
            ((c_h <= yx_loc[:, 0]) & (yx_loc[:, 0] < (c_h + self.crop_size))) & (
                        (c_w <= yx_loc[:, 1]) & (yx_loc[:, 1] < (c_w + self.crop_size))))
        yx_loc = yx_loc[loc_selection] - [c_h, c_w]

        noc_points = noc_points[loc_selection]

        # mask_image = np.zeros((self.crop_size, self.crop_size, 1))
        # mask_image[yx_loc[:, 0], yx_loc[:, 1], 0] = 1

        # noc_image = np.ones((self.crop_size, self.crop_size, 3)) * -1
        # noc_image[yx_loc[:, 0], yx_loc[:, 1], :] = noc_points
        num_points = np.array(yx_loc.shape[0])
        # num_points[num_points == 0] = 1
        data_dict['num_points'] = torch.from_numpy(num_points)
        # data_dict['mask_image'] = torch.from_numpy(mask_image.transpose([2, 0, 1]))
        # data_dict['noc_image'] = torch.from_numpy(noc_image.transpose([2, 0, 1]))

        total_pad = self.max_pad - num_points

        pad_width = ((0, total_pad), (0, 0))

        noc_points = np.pad(noc_points, pad_width=pad_width, mode='constant', constant_values=0)
        yx_loc = np.pad(yx_loc, pad_width=pad_width, mode='constant', constant_values=0)

        data_dict['noc_points'] = torch.from_numpy(noc_points)
        data_dict['yx_loc'] = torch.from_numpy(yx_loc)

        data_dict['image'] = self.transform(data_dict['image'])

        return data_dict

    def __len__(self):
        return self.len
        # return 1
