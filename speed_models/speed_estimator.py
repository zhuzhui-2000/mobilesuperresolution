from .SpeedModel import block_b
import torch
import torch.nn as nn
from models.ops import BinaryConv2d, rounding
from loss_config import mobile_device, compute_device
import numpy as np

class BlockBSpeedEstimator(nn.Module):

    def __init__(self, type, mobile_device=mobile_device, compute_device=compute_device, scale=2):
        super(BlockBSpeedEstimator, self).__init__()
        self.estimator = block_b(mobile_device, compute_device, scale, 3).eval()  # load pretrained model
        self.type = type

    def forward(self, x):

        if self.type == 'channel':
            return self.estimateByModuleChannel(x.body)
        elif self.type == 'tensor':
            return self.estimateByChannelNum(x)
        elif self.type == 'mask':
            return self.estimateByMask(x)

    @torch.no_grad()
    def estimateByModuleChannel(self, module: nn.Module):
        channels = []
        for m in module.children():
            if isinstance(m, nn.Conv2d) and not isinstance(m, BinaryConv2d):
                channels.append(m.in_channels)
        channels.append(channels[0])
        input_channels = torch.tensor(channels, dtype=torch.float).cuda()
        print(input_channels.shape)
        output = self.estimator(input_channels)
        return output

    @torch.no_grad()
    def estimateByChannelNum(self, x):
        print("estimateByChannelNum")
        #TODO
        # output = self.estimator(x)
        output = (x[1]+0.2*x[0])*((x[2]*x[2])) / 40
        return output

    @torch.no_grad()
    def estimateByMask(self, module: nn.Module, block_mask: nn.Module):
        # channels = self.get_unmask_number(module.block_mask)
        channels = self.get_unmask_number(block_mask)
        for m in module.body.children():
            if isinstance(m, BinaryConv2d):
                channels = torch.cat([channels, self.get_unmask_number(m)])
        channels = torch.cat([channels, channels[0].unsqueeze(0)])
        print(channels.shape)
        output = self.estimator(channels)
        return output

    @torch.no_grad()
    def estimateByMyMask(self, module: nn.Module, block_mask: nn.Module):
        # channels = self.get_unmask_number(module.block_mask)
        channels = self.get_unmask_number(block_mask)
        
        # for m in module.body.children():
        #     if isinstance(m, BinaryConv2d):
        #         channels = torch.cat([channels, self.get_unmask_number(m)])
        channels = torch.cat([channels, self.get_unmask_number(module.split)])
        # channels = torch.cat([channels, channels[0].unsqueeze(0)])
        
        output = 0
        kernels = torch.Tensor([3,5,7]).cuda()

        #TODO: speed 预测有问题，先使用线性模型进行预测
        # for i in range(3):
        #     output = output + self.estimator(torch.cat([channels, kernels[i].unsqueeze(0)]))*module.alpha[i]

        for i in range(3):
            output = output + ((channels[1]+0.2*channels[0])*((kernels[i]*kernels[i]).unsqueeze(0))*module.alpha[i]) / 40
        return output


    @staticmethod
    def get_unmask_number(m):
        assert isinstance(m, BinaryConv2d), f'Get {m}'
        w = m.weight.detach()
        binary_w = rounding(w)
        return binary_w.sum().unsqueeze(0)
