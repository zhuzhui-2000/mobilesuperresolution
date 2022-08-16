from utils import LPIPS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import lpips

from models.ops import BinaryConv2d, rounding
from models.wdsr_b import AggregationLayer

from torch import Tensor

__all__ = [
    "SpeedLoss"
]


class SpeedLoss(nn.Module):

    def __init__(self, scale=1):
        super(SpeedLoss, self).__init__()
        self.scale = scale

    def forward(self, speed, target, gamma=0.01, method='clamp'):
        assert method in ['mse', 'clamp', 'abs']
        diff = (speed - target) / self.scale
        if method == 'clamp':
            loss = torch.clamp_min(diff, 0)
        elif method == 'mse':
            loss = diff ** 2
        elif method == 'abs':
            loss = torch.abs(diff)
        return loss.mean() * gamma
