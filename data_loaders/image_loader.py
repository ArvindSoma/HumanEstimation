import os
import numpy as np
import time
import cv2
import torch
import random
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from glob import glob


class ImageLoader(Dataset):
    def __init__(self,
                 train=False,
                 parent_dir='',
                 file_ext='png',
                 transform=transforms.Compose([
                     transforms.ToTensor()])):
        # ,
        #                      transforms.Normalize(mean=(0, 0, 0), std=(0.5, 0.5, 0.5))

        self.parent_dir = parent_dir
        self.train = train
        self.transform = transform
        self.max_pad = 1200

        self.data = sorted(glob(os.path.join(parent_dir, '*.{}'.format(file_ext))))

        self.len = len(self.data)

    def __getitem__(self, index):
        """
        :param index: Index value
        :return: Dictionary of data points
        """

        data_dict = {}

        data_point = self.data[index]
        # internal_file_loc = data_point['file_name']

        image = cv2.imread(filename=self.data[index])

        image = cv2.resize(image, (256, 256), cv2.INTER_CUBIC)
        data_dict['image'] = self.transform(image)

        return data_dict

    def __len__(self):
        return self.len
