import os
import numpy as np
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
                 point_select=0.8,
                 flip=True,
                 transform=transforms.Compose([transforms.ToTensor()])):

        self.parent_dir = parent_dir
        self.base_size = base_size
        self.scales = scales
        self.crop_size = crop_size
        self.flip = flip
        self.point_select = point_select
        self.train = train
        self.transform = transform

        if self.train:
            read_file_name = '../data/train.pkl'

        else:
            read_file_name = '../data/valminusminival.pkl'
            self.point_select = 1

        with open(read_file_name, 'rb') as read_file:
            self.data = pickle.load(read_file)

        self.len = len(self.data)

    def fixed_rescale_crop(self, image, center=True):
        w, h, c = image.shape
        center = not self.train
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)

        img = cv2.resize(image, (oh, ow), cv2.INTER_CUBIC)
        w, h, _ = img.shape
        if center:
            x1 = int(round((w - self.crop_size) / 2.))
            y1 = int(round((h - self.crop_size) / 2.))
        else:
            x1 = random.randint(0, w - self.crop_size)
            y1 = random.randint(0, h - self.crop_size)
        img = img[x1: x1 + self.crop_size, y1: y1 + self.crop_size, :]

        return img, (ow, oh), (x1, y1)

    def __getitem__(self, index):
        """

        :param index: Index value
        :return: Dictionary of data points
        """

        data_dict = {}

        data_point = self.data[index]
        internal_file_loc = data_point['file_name']

        image = cv2.imread(filename=os.path.join(self.parent_dir, internal_file_loc))
        w, h, c = image.shape
        xy_loc = data_point['xy'][:, [1, 0]]
        point_len, _ = xy_loc.shape
        noc_points = data_point['noc']

        xy_loc = np.random.choice(xy_loc, (point_len * self.point_select), replace=False)

        data_dict['image'], (w_, h_), (c_w, c_h) = self.fixed_rescale_crop(image=image)
        xy_loc = (xy_loc / [w, h] * [w_, h_] - [c_w, c_h]).astype('int')

        mask_image = np.zeros((self.crop_size, self.crop_size, 1))
        mask_image[xy_loc[0], xy_loc[1], 0] = 1

        noc_image = np.zeros((self.crop_size, self.crop_size, 3))
        noc_image[xy_loc[0], xy_loc[1], :] = noc_points

        data_dict['mask_image'] = torch.from_numpy(mask_image.transpose([2, 0, 1]))
        data_dict['noc_image'] = torch.from_numpy(noc_image.transpose([2, 0, 1]))

        data_dict['image'] = self.transform(data_dict['image'])

        return data_dict

    def __len__(self):
        return self.len
