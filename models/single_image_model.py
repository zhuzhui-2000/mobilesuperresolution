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

from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import random

warnings.simplefilter("ignore", UserWarning)
class Result_Model(nn.Module):

    def __init__(self, scale, channel=48,blocks=7, kernel=3):
        super(Result_Model, self).__init__()

        self.image_mean = 0.5
        kernel_size = 3
        skip_kernel_size = 5
        weight_norm = torch.nn.utils.weight_norm
        num_inputs = 3
        self.scale = scale
        self.IN = channel

        num_outputs = scale * scale * num_inputs

        
        
        self.encoder = weight_norm(
            nn.Conv2d(
                num_inputs,
                self.IN,
                kernel_size,
                padding=kernel_size // 2))
        
        body = []
        for block in range(blocks):
            body.append(Block(
                IN=channel,
                split=channel,
                kernel_size=kernel,
                weight_norm=weight_norm)
            )

        conv = weight_norm(
            nn.Conv2d(
                self.IN,
                channel,
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
        
        shuf.append(nn.ConvTranspose2d(channel,3,5,stride=scale))


        self.shuf = nn.Sequential(*shuf)

            # shuf.append(nn.Conv2d(channel,channel*4,3,stride=1,padding=1))
            # shuf.append(nn.PixelShuffle(2))
            # shuf.append(nn.LeakyReLU())
            # shuf.append(nn.Conv2d(channel,channel*4,3,stride=1,padding=1))
            # shuf.append(nn.PixelShuffle(2))
            # shuf.append(nn.LeakyReLU())
            # shuf.append(nn.Conv2d(channel,3,3,stride=1,padding=1))
        
        
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        

    def forward(self, x, height=1080,weight=1920):
        
        B, N, C, H, W = x.shape
        out_list=[]
        for i in range(N):
            x_in = x[:,i,:,:,:]
            x_ = self.encoder(x_in)
            # base = nn.functional.interpolate(x_in, scale_factor=4, mode='bilinear', align_corners=False)
            # x_in = x_ - self.image_mean
            x_ = self.body(x_) + x_
            x_ = self.shuf(x_) 
            
            x_ = nn.functional.interpolate(x_,size=(height,weight),mode='bilinear')
            
            out_list.append(x_)
        out = torch.stack(out_list,dim=1)

        return out


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
        

        x = self.body(x) + x

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