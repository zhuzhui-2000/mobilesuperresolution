import warnings
import argparse
import importlib
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.utils import save_image

import common.meters
import common.modes
import common.metrics

from utils.estimate import test
from utils import logging_tool
from collections import OrderedDict

import models
from utils import attr_extractor, loss_printer
from loss_config import update_weight
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import random

warnings.simplefilter("ignore", UserWarning)
class Result_Model(nn.Module):

    def __init__(self, scale, filename):
        super(Result_Model, self).__init__()

        self.image_mean = 0.5
        kernel_size = 3
        skip_kernel_size = 5
        weight_norm = torch.nn.utils.weight_norm
        num_inputs = 3
        self.scale = scale
        self.idx = self.file_reader(filename)
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
        if scale == 4:
            shuf.append(nn.Conv2d(IN,IN*4,3,stride=1,padding=1))
            shuf.append(nn.PixelShuffle(2))
            shuf.append(nn.LeakyReLU())
            shuf.append(nn.Conv2d(IN,IN*4,3,stride=1,padding=1))
            shuf.append(nn.PixelShuffle(2))
            shuf.append(nn.LeakyReLU())
            shuf.append(nn.Conv2d(IN,3,3,stride=1,padding=1))
        self.shuf = nn.Sequential(*shuf)
        
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        t=time.time()
        B, N, C, H, W = x.shape
        x_list = []
        for image in range(N):
            x_in = x[:,image,:,:,:]
            base = nn.functional.interpolate(x_in, scale_factor=4, mode='bilinear', align_corners=False)
            # x_in = x_ - self.image_mean
            x_ = self.body(x_in) 
            x_ = self.shuf(x_) + base
            x_list.append(x_)
        print((time.time()-t)/N)
        return torch.stack(x_list,dim=1)

    def file_reader(self, filename):
        with open(filename, 'r') as f:
            status = eval(f.readlines()[-1].replace('\n', ''))[1]
        self.IN = status[0][0]
        print(status)
        return status


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
            conv = weight_norm(nn.Conv2d(input_dim, input_dim, kernal_size, padding = kernal_size // 2, groups = input_dim))
            body.append(conv)
            body.append(nn.ReLU(inplace=True))
            conv = weight_norm(nn.Conv2d(input_dim, output_dim, 1, padding=1 // 2,))
        else:
            conv = weight_norm(nn.Conv2d(input_dim, output_dim, kernal_size, padding=kernal_size // 2))
            body.append(conv)
            body.append(nn.ReLU(inplace=True))
            conv = weight_norm(nn.Conv2d(input_dim, output_dim, kernal_size, padding=kernal_size // 2))
            body.append(conv)
        
        self.body = nn.Sequential(*body)

    def forward(self, x):
        x = self.body(x)
        return x