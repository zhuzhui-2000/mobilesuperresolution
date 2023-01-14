from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from os import sep

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from models.ops import BinaryConv2d, rounding

try:
    from speed_models import BlockBSpeedEstimator
except ImportError:
    pass

from collections import namedtuple

ModelOutput = namedtuple(
    "ModelOutput",
    "sr speed_accu speed_curr"
)

__all__ = ['Result_Model', ]


class Result_Model(nn.Module):

    def __init__(self, scale, blocks, idx):
        super(Result_Model, self).__init__()
        self.image_mean = 0.5
        kernel_size = 3
        skip_kernel_size = 5
        weight_norm = torch.nn.utils.weight_norm
        num_inputs = 3
        self.scale = scale
        self.idx = idx
        self.blocks = blocks
        self.IN = self.idx[0][0]

        num_outputs = scale * scale * num_inputs

        body = []
        conv = weight_norm(
            nn.Conv2d(
                num_inputs,
                self.IN,
                kernel_size,
                padding=kernel_size // 2))
        body.append(conv)

        for block in self.idx:
            IN = block[0]
            split = block[1]
            kernel_size = block[2]
            body.append(Block(
                IN=IN,
                split=split,
                kernel_size=kernel_size,
                weight_norm=weight_norm)
            )

        conv = weight_norm(
            nn.Conv2d(
                self.IN,
                num_outputs,
                kernel_size,
                padding=kernel_size // 2))

        body.append(conv)
        self.body = nn.Sequential(*body)
        conv = weight_norm(
            nn.Conv2d(
                num_inputs,
                num_outputs,
                skip_kernel_size,
                padding=skip_kernel_size // 2))

        self.skip = conv

        shuf = []
        if scale > 1:
            shuf.append(nn.PixelShuffle(scale))
        self.shuf = nn.Sequential(*shuf)

    def forward(self, x):
        x = x - self.image_mean
        x = self.body(x) + self.skip(x)
        x = self.shuf(x)
        return x


class Block(nn.Module):
    def __init__(self, IN, split, kernel_size, weight_norm=torch.nn.utils.weight_norm):
        super(Block, self).__init__()
        self.split = IN-split
        self.IN = IN
        self.conv_channel = split
        body = []

        conv = Conv_sep(self.conv_channel, self.conv_channel, kernel_size)
        body.append(conv)


        self.body = nn.Sequential(*body)

    def forward(self, x):
        print(x.shape)
        if(self.split > 0):
            x_1, x = torch.split(x,[self.split,self.conv_channel],dim=1)
        x = self.body(x) + x
        if(self.split > 0):
            x = torch.cat((x_1,x),dim=1)
        return x

class Conv_sep(nn.Module):
    def __init__(self, input_dim, output_dim, kernal_size,weight_norm=torch.nn.utils.weight_norm, seperate=False):
        super(Conv_sep,self).__init__()
        self.seperate = seperate
        self.kernel_size = kernal_size
        
        body = []
        if self.seperate :
            conv = weight_norm(nn.Conv2d(input_dim, input_dim, kernal_size, padding = kernal_size // 2, groups = input_dim))
            body.append(conv)
            body.append(nn.ReLU(inplace=True))
            conv = weight_norm(nn.Conv2d(input_dim, output_dim, 1, padding=1 // 2,))
            body.append(conv)
        else:
            conv = weight_norm(nn.Conv2d(input_dim, output_dim, kernal_size, padding=kernal_size // 2))
            body.append(conv)
            body.append(nn.ReLU(inplace=True))
        
        self.body = nn.Sequential(*body)

    def forward(self, x):
        x = self.body(x)
        return x
