"""
Commonly used functions
"""

import torch
from torchvision import utils


# Prep models for train
def write_image(writer, sample, name, niter, n_row=4, seg_mask=False):
    """

    :param writer: Torch writer
    :param sample: 4D Tensors in range [-1, 1]
    :param name: Name of category
    :param niter:
    :param n_row: Number of images to display per row
    :param seg_mask:
    :return:
    """

    permute = [2, 1, 0]

    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
                             (24, 255, 128), (255, 24, 128), (255, 128, 24), (128, 255, 24)])

    if seg_mask is True:
        segment_mask = torch.zeros_like(sample).long()

        for idx in range(25):
            segment_mask[0, :, :][sample[0, :, :] == idx] = torch.tensor(label_colors[idx][0]).long()
            segment_mask[1, :, :][sample[0, :, :] == idx] = torch.tensor(label_colors[idx][1]).long()
            segment_mask[2, :, :][sample[0, :, :] == idx] = torch.tensor(label_colors[idx][2]).long()

        writer.add_image(name, utils.make_grid(segment_mask, nrow=n_row), niter)
    else:
        sample = (sample + 1) / 2
        writer.add_image(name, utils.make_grid(sample, nrow=n_row, pad_value=0.05)[permute, :, :], niter)

