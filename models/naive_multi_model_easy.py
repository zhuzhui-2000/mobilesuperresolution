import warnings
import argparse
import importlib
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
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
from mmedit.models.backbones.sr_backbones.basicvsr_net import (
    ResidualBlocksWithInputConv, SPyNet)
from mmedit.models.common import PixelShufflePack, flow_warp
import numpy as np
import random

warnings.simplefilter("ignore", UserWarning)
class Naive_model(nn.Module):

    def __init__(self, scale, filename, spynet_pretrained=None):
        super(Naive_model, self).__init__()

        self.image_mean = 0.5
        kernel_size = 3
        skip_kernel_size = 5
        weight_norm = torch.nn.utils.weight_norm
        num_inputs = 3
        self.scale = scale
        self.idx = self.file_reader(filename)
        self.IN = self.idx[0][0]
        self.flownet = SPyNet(pretrained=spynet_pretrained)
        for m in self.flownet.parameters():
            m.requires_grad=False
        num_outputs = scale * scale * num_inputs

        self.body = nn.ModuleDict()
        self.encode = weight_norm(
            nn.Conv2d(
                num_inputs,
                self.IN,
                kernel_size,
                padding=kernel_size // 2))
        
        idx_list = [x for x in range(len(self.idx))]
        for idx,block in enumerate(self.idx):
            if idx==0:

                IN = block[0]*2+2
                OUT = block[0]
                split = block[1]
                kernel_size = block[2]
                self.body[str(idx)]=(Block(
                    IN=IN,
                    OUT=OUT,
                    split=split,
                    kernel_size=kernel_size,
                    weight_norm=weight_norm)
                )
            else:
                IN = block[0]
                OUT = block[0]
                split = block[1]
                kernel_size = block[2]
                self.body[str(idx)]=(Block(
                    IN=IN,
                    OUT=OUT,
                    split=split,
                    kernel_size=kernel_size,
                    weight_norm=weight_norm)
                )

        self.decode = weight_norm(
            nn.Conv2d(
                self.IN,
                num_outputs,
                kernel_size,
                padding=kernel_size // 2))


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

        B, N, C, H, W = x.shape
        lqs_1 = x[:, :-1, :, :, :].reshape(-1, C, H, W)
        lqs_2 = x[:, 1:, :, :, :].reshape(-1, C, H, W)
        flows_forward = self.flownet(lqs_2, lqs_1).view(B, N - 1, 2, H, W)
        x_list = []
        pre_feats = None
        for image in range(N):
            x_ = x[:,image,:,:,:]
            res = x_
            x_ = self.encode(x_)
            for idx in range(len(self.idx)):
                
                if idx==0:
                    if (image==0):
                        x_warp = x_
                        flow = torch.zeros_like(flows_forward[:,0,:,:,:]).to('cuda')
                        pre_feats=x_

                    else:
                        x_pre_feat = pre_feats
                        pre_feats=x_
                        #x_pre_feat = F.interpolate(x_list[image-1],size=(H,W),aligned_corner=True)
                        flow = flows_forward[:,image-1,:,:,:]
                        x_warp = flow_warp(x_pre_feat,flow.permute(0, 2, 3, 1))
                    x_c = torch.cat((flow,x_warp,x_),dim=1)
                else:
                    x_c = x_
                
                x_ = self.body[str(idx)](x_c)+x_

            base = nn.functional.interpolate(res, scale_factor=4, mode='bilinear', align_corners=False)
            x_ = self.decode(x_)
            
            x_ = self.shuf(x_) + base
            x_list.append(x_)
        return torch.stack(x_list,dim=1)

    def file_reader(self, filename):
        with open(filename, 'r') as f:
            status = eval(f.readlines()[-1].replace('\n', ''))[1]
        self.IN = status[0][0]
        print(status)
        return status


class Block(nn.Module):
    def __init__(self, IN, OUT, split, kernel_size, weight_norm=torch.nn.utils.weight_norm):
        super(Block, self).__init__()
        self.split = IN-split
        self.IN = IN
        self.OUT = OUT
        self.conv_channel = split
        body = []

        conv = nn.Conv2d(IN, OUT, kernel_size, padding = kernel_size // 2)
        body.append(conv)
        body.append(nn.ReLU(inplace=True))
        conv = nn.Conv2d(OUT, OUT, kernel_size, padding = kernel_size // 2)
        body.append(conv)
        
        self.skip = nn.Conv2d(2*self.IN, self.IN, 1, padding = 0)

        self.body = nn.Sequential(*body)

    def forward(self, x):
        
        # if(self.split > 0):
        #     x_1, x = torch.split(x,[self.split,self.conv_channel],dim=1)
        x = self.body(x) 
        # if(self.split > 0):
        #     x = torch.cat((x_1,x),dim=1)
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