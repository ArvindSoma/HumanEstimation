"""
Basic Blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvison
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from collections import namedtuple


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, filter=4, stride=1, dilation=1, pad=1, bn=None):
        super(ConvLayer, self).__init__()
        seq1 = [nn.Conv2d(in_ch, out_ch, filter, stride=stride, padding=pad, dilation=dilation)]
        if bn is not None:
            seq1 += [bn(out_ch)]
        else:
            raise Exception('Normalization not declared!')

        self.sequence1 = nn.Sequential(*seq1)

    def forward(self, x):
        net = self.sequence1(x)
        return net


class UpConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, size, stride=1, bn=None, dropout=False, skip=True):
        super(UpConvLayer, self).__init__()
        seq1 = [nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, 4, stride=stride, padding=1))]
        if bn is not None:
            seq1 += [bn(out_ch)]
        else:
            raise Exception('Normalization not declared!')

        self.sequence1 = nn.Sequential(*seq1)

        self.leaky_relu = nn.LeakyReLU(0.2)
        seq2 = []
        if skip:
            for idx in range(2):
                seq2 += [ConvLayer(out_ch * 2, out_ch * 2, filter=3, bn=bn)]
                if idx == 0:
                    seq2 += [nn.LeakyReLU()]

        else:
            for idx in range(2):
                seq2 += [ConvLayer(out_ch, out_ch, filter=3, bn=bn)]
                if idx == 0:
                    seq2 += [nn.LeakyReLU()]

        self.sequence2 = nn.Sequential(*seq2)
        self.dropout = dropout

    def forward(self, x, skip_net=None):

        net = self.sequence1(x)

        if self.dropout:
            net = F.dropout(net, 0.2)

        if skip_net is not None:
            net = torch.cat([net, self.skip_net], 1)

        net = self.sequence2(self.leaky_relu(net))

        return net


class MultiDilation(nn.Module):
    def __init__(self, dim_out, norm_layer=nn.BatchNorm2d, dilation=2):
        super(MultiDilation, self).__init__()

        # self.modules = nn.ModuleList()
        dil = dilation
        # for dil in range(1, dilation + 1):
        self.seq = nn.Sequential(
            ConvLayer(dim_out, dim_out, dilation=dil, filter=3, pad=dil, bn=norm_layer),
            nn.LeakyReLU(0.2),
            ConvLayer(dim_out, dim_out, dilation=dil, filter=3, pad=dil, bn=norm_layer),
        )
            # self.modules.append(seq)

    def forward(self, x):
        # outputs = 0
        # for module in self.modules:
        #     outputs += module(x)
        outputs = self.seq(x)
        return outputs


class ResNext50(nn.Module):
    def __init__(self):
        super(ResNext50, self).__init__()
        model = torch.hub.load('pytorch/vision:v0.4.0', 'resnext50_32x4d', pretrained=True)
        model.eval()
        model = nn.Sequential(*model.children())
        layer_list = [3, 5, 6, 7]

        out = namedtuple("out", ['layer_' + str(i + 1) for i in range(4)])
        start = 0
        layer_tuple = tuple()

        for idx, val in enumerate(layer_list):
            layer_tuple += (nn.Sequential(model[start: val]),)
            start = val

        self.layers = out(layer_tuple)
        self.conv_1 = nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0)
        self.conv_2 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)
        self.conv_3 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.interpolate = nn.Upsample(scale_factor=2)

    def forward(self, x):
        l1 = self.layers.layer_1(x)
        l2 = self.layers.layer_2(l1)
        l3 = self.layers.layer_3(l2)
        l4 = self.layers.layer_4(l3)

        out_3 = self.conv_2(l3) + self.interpolate(self.conv_1(l4))
        out_2 = self.conv_1(l2) + self.interpolate(out_3)
        out_1 = l1 + self.interpolate(out_2)

        return out_1


class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet18_fpn = resnet_fpn_backbone(backbone_name='resnet18', pretrained=True)

    def forward(self, x):
        return self.resnet18_fpn(x)