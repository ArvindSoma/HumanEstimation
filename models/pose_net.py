"""
Pose Estimation Network
"""

from models.basic import *


class PoseNet(nn.Module):
    def __init__(self, latent=''):
        super(PoseNet, self).__init__()
