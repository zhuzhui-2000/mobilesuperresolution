import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class BinaryConv2d(nn.Conv2d):
    """docstring for QuanConv"""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, least_channel=8):
        super(BinaryConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias)
        init.uniform_(self.weight, 0.5, 1)
        self.least_channel=least_channel
        # init.constant_(self.weight, 1)

    def forward(self, x, y=None):
        w = self.weight.detach()
        # binary_w = (w >= 0.5).float()
        binary_w = rounding(w, self.least_channel)
        residual = w - binary_w
        weight = self.weight - residual

        output = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output

    def init(self, value=0.5):
        init.constant_(self.weight, value)
        # init.normal_(self.weight, value, std=0.001)


def rounding(weight, least_channel=8):
    w = (weight >= 0.5).float()
    if least_channel>0:
        v, idx = torch.topk(weight, least_channel, dim=0)
        w_4 = (weight >= v[-1]).float()
        if torch.sum(w) >= least_channel:
            return w
        else:
            return w_4
    else:
        return w
