"""
Texture Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Texture(nn.Module):
    def __init__(self, n_feature, dimensions, n_textures=0, n_level=4):
        super(Texture, self).__init__()
        # checker_board = np.kron([[1, 0] * 4, [0, 1] * 4] * 4, np.ones((2 * dimensions, dimensions)))
        # checkerboard = torch.Tensor(checker_board)
        self.n_level = n_level
        self.n_textures = n_textures

        if self.n_level > 1:
            self.register_parameter('feature_map', nn.Parameter(
                2 * torch.randn((n_textures, n_feature, 2 * dimensions, dimensions), requires_grad=True) - 1.0))

        else:
            self.register_parameter('feature_map', nn.Parameter(
                2 * torch.randn((n_textures, n_feature, dimensions, dimensions), requires_grad=True) - 1.0))

        self.first_dim = dimensions

    def forward(self, uv_input, texture_id=0, n_batch=1):

        _, uv_h, uv_w, c = uv_input.shape
        feature_select = self.feature_map.repeat(n_batch, 1, 1, 1)
        uv_input = uv_input.view(n_batch * self.n_textures, uv_h, uv_w, 2)
        dim = self.first_dim
        start = 0
        level_mapped_feature = 0
        for idx in range(self.n_level):

            feature_level = feature_select[:, :, start:(start + dim), :dim]
            level_mapped_feature += torch.nn.functional.grid_sample(feature_level, uv_input,
                                                                    mode='bilinear', padding_mode='border')

            start += dim
            dim = dim // 2

        return level_mapped_feature
