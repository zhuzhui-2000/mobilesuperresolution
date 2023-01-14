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

__all__ = ['NAS_MODEL', ]


class NAS_MODEL(nn.Module):

    def __init__(self, params):
        super(NAS_MODEL, self).__init__()
        self.image_mean = params.image_mean
        kernel_size = 3
        skip_kernel_size = 5
        weight_norm = torch.nn.utils.weight_norm
        num_inputs = params.num_channels
        scale = params.scale
        self.scale = scale
        self.num_blocks = params.num_blocks
        self.num_residual_units = params.num_residual_units
        self.remain_blocks = params.num_blocks
        self.width_search = params.width_search
        self.idx_kernel = [3,5,7]
        # self.bottle = params.bottleneck_type
        # self.seperate = params.seperate

        num_outputs = scale * scale * params.num_channels

        conv = weight_norm(
            nn.Conv2d(
                num_inputs,
                params.num_residual_units,
                kernel_size,
                padding=kernel_size // 2))

        init.ones_(conv.weight_g)
        init.zeros_(conv.bias)
        self.head = conv
        self.speed_estimator = BlockBSpeedEstimator('mask' if params.width_search else 'channel').eval()

        body = nn.ModuleList()
        for _ in range(params.num_blocks):
            body.append(MyAggregationLayer(
                num_residual_units=params.num_residual_units,
                kernel_size=kernel_size,
                weight_norm=weight_norm,
                res_scale=1 / math.sqrt(params.num_blocks),
                width_search=params.width_search)
            )
        self.body = body

        if self.width_search:
            self.mask = BinaryConv2d(in_channels=params.num_residual_units,
                                          out_channels=params.num_residual_units,
                                          groups=params.num_residual_units)
            # self.mask.init(0.5)
        conv = weight_norm(
            nn.Conv2d(
                params.num_residual_units,
                num_outputs,
                kernel_size,
                padding=kernel_size // 2))
        init.ones_(conv.weight_g)
        init.zeros_(conv.bias)
        self.tail = conv

        conv = weight_norm(
            nn.Conv2d(
                num_inputs,
                num_outputs,
                skip_kernel_size,
                padding=skip_kernel_size // 2))
        init.ones_(conv.weight_g)
        init.zeros_(conv.bias)
        self.skip = conv

        shuf = []
        if scale > 1:
            shuf.append(nn.PixelShuffle(scale))
        self.shuf = nn.Sequential(*shuf)

        if params.pretrained:
            self.load_pretrained()

    def forward(self, x):
        x = x - self.image_mean
        y = self.head(x)
        speed_accu = x.new_zeros(1) #TODO：这个地方它没有考虑shuf和skip的时延，或者说他没有考虑基本框架的时延，这个需要改一改哈
        for module in self.body:
            if self.width_search:
                speed_curr = self.speed_estimator.estimateByMyMask(module, self.mask)
            else:
                speed_curr = self.speed_estimator.estimateByChannelNum(module)
            y = self.mask(y)
            y, speed_accu = module(y, speed_curr, speed_accu)
        if self.width_search:
            y = self.mask(y)
        y = self.tail(y) + self.skip(x)
        y = self.shuf(y)
        out = y + self.image_mean
        # batch_size, channels, in_height, in_width = y.size()
        
        # channels = channels / (self.scale * self.scale)
        # out = torch.split(y, self.scale * self.scale, dim=1)
        # out = [torch.reshape(a,(batch_size,1,self.scale,self.scale,in_height,in_width)) for a in out]
        # out = [torch.transpose(a,4,2) for a in out]
        # out = [torch.transpose(a,4,3) for a in out]
        # out = [torch.transpose(a,4,5) for a in out]
        # out = [torch.reshape(a,(batch_size,1,self.scale*in_height,self.scale*in_width)) for a in out]
        # out = torch.cat(out,1)

        # out = out + self.image_mean
        

        return out, speed_accu

    @torch.no_grad()
    def get_current_blocks(self):
        num_blocks = 0
        for module in self.body.children():
            if isinstance(module, MyAggregationLayer):
                if module.alpha1 < module.alpha2:  # not skip
                    num_blocks += 1
        return int(num_blocks)

    @torch.no_grad()
    def get_block_status(self):
        remain_block_idx = []
        for idx, module in enumerate(self.body.children()):
            
            if isinstance(module, MyAggregationLayer):
                alpha1, alpha2 = F.softmax(torch.stack([module.alpha1, module.alpha2], dim=0), dim=0)
                if alpha1 < alpha2:  # not skip
                    remain_block_idx.append(idx)
        return remain_block_idx

    @torch.no_grad()
    def get_width_from_block_idx(self, remain_block_idx):
        @torch.no_grad()
        def _get_width_from_weight(w):
            return int(rounding(w).sum())

        def _get_width_from_split(w,w_mask):
            return int((rounding(w_mask) * rounding(w)).sum())

        all_width = []
        for idx, module in enumerate(self.body.children()):
            width = []
            if idx in remain_block_idx and isinstance(module, MyAggregationLayer):
                # width.append(_get_width_from_weight(module.block_mask.weight))
                width.append(_get_width_from_weight(self.mask.weight))
                
                width.append(_get_width_from_split(module.split.weight,self.mask.weight))

                
                max_value, max_index = torch.max(module.alpha,0)
                
                best_kernel = self.idx_kernel[max_index]
                width.append(best_kernel)
                all_width.append(width)
        return all_width

    @torch.no_grad()
    def get_alpha_grad(self):
        for module in self.body.children():
            if isinstance(module, MyAggregationLayer):
                return module.alpha1.grad, module.alpha2.grad

    @torch.no_grad()
    def get_alpha(self):
        for module in self.body.children():
            if isinstance(module, MyAggregationLayer):
                return module.alpha1, module.alpha2

    @torch.no_grad()
    def length_grad(self, flag=False):
        for module in self.body.children():
            if isinstance(module, MyAggregationLayer):
                module.alpha1.requires_grad = flag
                module.alpha2.requires_grad = flag
                module.beta1.requires_grad = flag
                module.beta2.requires_grad = flag

    @torch.no_grad()
    def mask_grad(self, flag=False):
        for module in self.body.children():
            if isinstance(module, MyAggregationLayer):
                # module.block_mask.weight.requires_grad = flag
                    module.split.weight.requires_grad = flag
        self.mask.weight.requires_grad = flag

    @torch.no_grad()
    def kernel_grad(self, flag=False):
        for module in self.body.children():
            if isinstance(module, MyAggregationLayer):
                # module.block_mask.weight.requires_grad = flag
                max_value, max_index = torch.max(module.alpha,0)
                temp = torch.zeros(3)
                temp[max_index] = 1
                module.alpha = temp
                module.alpha.requires_grad = flag
        
        pass

    @torch.no_grad()
    def get_mask_grad(self):
        return self.mask.weight.grad

    @torch.no_grad()
    def get_mask_weight(self):
        return self.mask.weight.data

    @torch.no_grad()
    def load_pretrained(self):
        import os
        path, filename = os.path.split(__file__)
        weight_path = f"{path}/pretrained_weights"
        state_dict = torch.load(f"{weight_path}/wdsr_b_x{self.scale}_{self.num_blocks}_{self.num_residual_units}.pt",
                                map_location='cpu')
        state_dict_iterator = iter(state_dict.items())
        load_name, load_param = next(state_dict_iterator)
        for p in self.parameters():
            if p.size() == load_param.size():
                p.data = load_param
                try:
                    load_name, load_param = next(state_dict_iterator)
                except StopIteration:
                    pass


class Block(nn.Module):

    def __init__(self,
                 num_residual_units,
                 kernel_size,
                 weight_norm=torch.nn.utils.weight_norm,
                 res_scale=1,
                 width_search=False):
        super(Block, self).__init__()
        body = []
        expand = 6
        linear = 0.84

        # if width_search:
        #     # mask for first layer
        #     self.block_mask = BinaryConv2d(in_channels=num_residual_units,
        #                                    out_channels=num_residual_units,
        #                                    groups=num_residual_units)
        #     self.block_mask.init(0.5)
        conv = weight_norm(
            nn.Conv2d(
                num_residual_units,
                int(num_residual_units * expand),
                1,
                padding=1 // 2))
        init.constant_(conv.weight_g, 2.0)
        init.zeros_(conv.bias)

        body.append(conv)
        body.append(nn.ReLU(inplace=True))
        if width_search:
            # mask for second layer
            body.append(BinaryConv2d(in_channels=int(num_residual_units * expand),
                                     out_channels=int(num_residual_units * expand),
                                     groups=int(num_residual_units * expand)))

        conv = weight_norm(
            nn.Conv2d(
                num_residual_units * expand,
                int(num_residual_units * linear),
                1,
                padding=1 // 2))
        init.constant_(conv.weight_g, 2.0)
        init.zeros_(conv.bias)
        body.append(conv)

        if width_search:
            # mask for third layer
            body.append(BinaryConv2d(in_channels=int(num_residual_units * linear),
                                     out_channels=int(num_residual_units * linear),
                                     groups=int(num_residual_units * linear)))
        conv = weight_norm(
            nn.Conv2d(
                int(num_residual_units * linear),
                num_residual_units,
                kernel_size,
                padding=kernel_size // 2))
        init.constant_(conv.weight_g, res_scale)
        init.zeros_(conv.bias)
        body.append(conv)

        self.body = nn.Sequential(*body)

    def forward(self, x):
        # x = self.block_mask(x)
        x = self.body(x) + x
        return x


class AggregationLayer(Block):
    def __init__(self, **kwargs):
        super(AggregationLayer, self).__init__(**kwargs)

        # Skip
        self.alpha1 = nn.Parameter(data=torch.empty(1), requires_grad=True)
        self.beta1 = nn.Parameter(data=torch.zeros(1), requires_grad=True)
        init.uniform_(self.alpha1, 0, 0.2)

        # Preserve
        self.alpha2 = nn.Parameter(data=torch.empty(1), requires_grad=True)
        self.beta2 = nn.Parameter(data=torch.ones(1), requires_grad=True)
        init.uniform_(self.alpha2, 0.8, 1)

    def forward(self, x, speed_curr, speed_accu):
        # model_output = input
        # x = input.sr
        # speed_accu = input.speed_accu
        #
        # speed_curr = input.speed_curr
        if self.training:
            # self.alpha1.data, self.alpha2.data = F.gumbel_softmax(torch.stack([self.alpha1.data, self.alpha2.data],
            #                                                                   dim=0), dim=0, hard=False)
            # self.alpha1.data, self.alpha2.data = F.softmax(torch.stack([self.alpha1.data, self.alpha2.data],
            #                                                            dim=0), dim=0)
            # Get skip result
            sr1 = x
            # Get block result
            sr2 = self.body(x) + x

            beta1, beta2 = ConditionFunction.apply(self.alpha1, self.alpha2, self.beta1, self.beta2)
            self.beta1.data, self.beta2.data = beta1, beta2
            x = beta1 * sr1 + beta2 * sr2
            # model_output.speed_accu = beta2 * speed_curr + speed_accu
            speed_accu = beta2 * speed_curr + speed_accu
            return x, speed_accu
        else:
            if self.alpha1 >= self.alpha2:
                pass
            else:
                x = self.body(x) + x
            # model_output.speed_accu = speed_accu + self.beta2 * speed_curr
            speed_accu = speed_accu + self.beta2 * speed_curr
            return x, speed_accu

    def get_num_channels(self):
        channels = []
        for m in self.body.children():
            if isinstance(m, nn.Conv2d):
                channels.append(m.in_channels)
        channels.append(channels[0])
        return channels

class Conv_sep(nn.Module):
    def __init__(self, input_dim, output_dim, kernal_size,weight_norm=torch.nn.utils.weight_norm, seperate=False):
        super(Conv_sep,self).__init__()
        self.seperate = seperate
        self.kernel_size = kernal_size
        
        body = []
        if self.seperate :
            conv = weight_norm(nn.Conv2d(input_dim, input_dim, kernal_size, padding = kernal_size // 2, groups = input_dim))
            # init.constant_(conv.weight_g, 2.0)
            # init.zeros_(conv.bias)
            body.append(conv)
            body.append(nn.ReLU(inplace=True))
            conv = weight_norm(nn.Conv2d(input_dim, output_dim, 1, padding=1 // 2,))
            # init.constant_(conv.weight_g, 2.0)
            # init.zeros_(conv.bias)
            body.append(conv)
        else:
            conv = weight_norm(nn.Conv2d(input_dim, output_dim, kernal_size, padding=kernal_size // 2))
            # init.constant_(conv.weight_g, 2.0)
            # init.zeros_(conv.bias)
            body.append(conv)
        
        self.body = nn.Sequential(*body)

    def forward(self, x):
        x = self.body(x)
        return x


class Split_Block(nn.Module):

    def __init__(self,
                 num_residual_units,
                 kernel_size,
                 weight_norm=torch.nn.utils.weight_norm,
                 res_scale=1,
                 width_search=False,
                 block_type = 'normal', #block_type = 'inverted_bottle', 'bottle', 'normal'
                 seperate_type = True):  
        super(Split_Block, self).__init__()
        
        expand = 6
        linear = 0.84
        # Choose
        self.alpha = nn.Parameter(data=torch.ones(3), requires_grad=True)
        init.uniform_(self.alpha, 0.5, 1.5)
        self.beta = nn.Parameter(data=torch.zeros(3), requires_grad=True)

        self.split = BinaryConv2d(num_residual_units,num_residual_units,groups=num_residual_units, least_channel=0)
        # print(rounding(self.split.weight.detach()).sum())
        self.kernel_list = ['3','5','7']
        # if block_type == 'inverted_bottle':
        #     conv = weight_norm(
        #         nn.Conv2d(
        #             num_residual_units,
        #             int(num_residual_units * expand),
        #             1,
        #             padding=1 // 2))
        #     # init.constant_(conv.weight_g, 2.0)
        #     # init.zeros_(conv.bias)

        #     body.append(conv)
        #     body.append(nn.ReLU(inplace=True))
        #     if width_search:
        #         # mask for second layer
        #         body.append(BinaryConv2d(in_channels=int(num_residual_units * expand),
        #                                 out_channels=int(num_residual_units * expand),
        #                                 groups=int(num_residual_units * expand)))

        #     conv = Conv_sep(num_residual_units * expand, num_residual_units, kernel_size, seperate=seperate_type)
        #     body.append(conv)

        self.body=nn.ModuleDict()
        

        for kernel_ in self.kernel_list:
            body = []
            if block_type == 'normal':
                conv = Conv_sep(num_residual_units, num_residual_units, int(kernel_), seperate=seperate_type)
                body.append(conv)
                body.append(nn.ReLU(inplace=True))
            self.body[kernel_] = nn.Sequential(*body) 
            # self.body_list.append(self.body)


        # if block_type == 'bottle':
        #     conv = weight_norm(
        #         nn.Conv2d(
        #             num_residual_units,
        #             int(num_residual_units * linear),
        #             1,
        #             padding=1 // 2))
        #     # init.constant_(conv.weight_g, 2.0)
        #     # init.zeros_(conv.bias)

        #     body.append(conv)
        #     body.append(nn.ReLU(inplace=True))
        #     if width_search:
        #         # mask for second layer
        #         body.append(BinaryConv2d(in_channels=int(num_residual_units * linear),
        #                                 out_channels=int(num_residual_units * linear),
        #                                 groups=int(num_residual_units * linear)))

        #     conv = Conv_sep(int(num_residual_units * linear), num_residual_units, kernel_size, seperate=seperate_type)
        #     body.append(conv)

    def forward_body(self, x):
        x_1 = self.split(x) 
        x_2 = x - x_1
        x_3 = torch.clone(x_2)

        pro = F.softmax(self.alpha)
        
        
        
        for i, kernel_ in enumerate(self.kernel_list):
            x_ = (self.body[kernel_](x_1))*pro[i]
            x_3 = x_3 + x_
        x_3 = x_3 + x_1
        x_cat = x_2 + self.split(x_3) 
        return x_cat

    def forward(self, x):
        # x = self.block_mask(x)

        return self.forward_body(x)

class MyAggregationLayer(Split_Block):
    def __init__(self, **kwargs):
        super(MyAggregationLayer, self).__init__(**kwargs)

        # Skip
        self.alpha1 = nn.Parameter(data=torch.empty(1), requires_grad=True)
        self.beta1 = nn.Parameter(data=torch.zeros(1), requires_grad=True)
        init.uniform_(self.alpha1, 0, 0.2)

        # Preserve
        self.alpha2 = nn.Parameter(data=torch.empty(1), requires_grad=True)
        self.beta2 = nn.Parameter(data=torch.ones(1), requires_grad=True)
        init.uniform_(self.alpha2, 0.8, 1)

    def forward(self, x, speed_curr, speed_accu):
        # model_output = input
        # x = input.sr
        # speed_accu = input.speed_accu
        #
        # speed_curr = input.speed_curr
        if self.training:
            # self.alpha1.data, self.alpha2.data = F.gumbel_softmax(torch.stack([self.alpha1.data, self.alpha2.data],
            #                                                                   dim=0), dim=0, hard=False)
            # self.alpha1.data, self.alpha2.data = F.softmax(torch.stack([self.alpha1.data, self.alpha2.data],
            #                                                            dim=0), dim=0)
            # Get skip result
            sr1 = x
            # Get block result
            sr2 = self.forward_body(x)

            beta1, beta2 = ConditionFunction.apply(self.alpha1, self.alpha2, self.beta1, self.beta2)
            self.beta1.data, self.beta2.data = beta1, beta2
            x = beta1 * sr1 + beta2 * sr2
            # model_output.speed_accu = beta2 * speed_curr + speed_accu
            speed_accu = beta2 * speed_curr + speed_accu
            return x, speed_accu
        else:
            if self.alpha1 >= self.alpha2:
                pass
            else:
                x = self.forward_body(x)
            # model_output.speed_accu = speed_accu + self.beta2 * speed_curr
            speed_accu = speed_accu + self.beta2 * speed_curr
            return x, speed_accu

    def get_num_channels(self):
        channels = []
        for m in self.body.children():
            if isinstance(m, nn.Conv2d):
                channels.append(m.in_channels)
        channels.append(channels[0])
        return channels

class AggregationFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sr1, sr2, speed_accu, speed_curr, alpha1, alpha2, beta1, beta2):
        ctx.num_outputs = 2
        with torch.no_grad():
            if alpha1 > alpha2:
                # when a1 >= a2
                beta1.data = beta1.new_ones(1)  # set b1 = 0
                beta2.data = beta2.new_zeros(1)  # set b2 = 1
            else:
                # when a1 < a2
                beta1.data = beta1.new_zeros(1)  # set b1 = 0
                beta2.data = beta2.new_ones(1)  # set b2 = 1

        ctx.save_for_backward(sr1, sr2, speed_accu, speed_curr, beta1, beta2)

        sr = sr1 * beta1 + sr2 * beta2
        total_speed = speed_curr * beta2 + speed_accu

        return sr, total_speed

    @staticmethod
    def backward(ctx, grad_output_sr, grad_output_speed):
        sr1, sr2, speed_accu, speed_curr, beta1, beta2 = ctx.saved_tensors
        grad_sr1 = grad_output_sr * beta1
        grad_sr2 = grad_output_sr * beta2
        grad_speed_accu = grad_output_speed * beta2
        grad_speed_curr = grad_output_speed
        grad_beta1 = grad_output_sr.bmm(beta1)  # for grad_alpha1
        grad_beta2 = grad_output_sr * beta2 + grad_output_speed * speed_curr  # for grad_alpha2

        grad_alpha1 = grad_beta1
        grad_alpha2 = grad_beta2

        return grad_sr1, grad_sr2, grad_speed_accu, grad_speed_curr, grad_alpha1, grad_alpha2, None, None


class ConditionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, alpha1, alpha2, beta1, beta2):
        with torch.no_grad():
            if alpha1 >= alpha2:
                # when a1 >= a2
                beta1.data = beta1.new_ones(1)  # set b1 = 0
                beta2.data = beta2.new_zeros(1)  # set b2 = 1
            else:
                # when a1 < a2
                beta1.data = beta1.new_zeros(1)  # set b1 = 0
                beta2.data = beta2.new_ones(1)  # set b2 = 1

        return beta1, beta2

    @staticmethod
    def backward(ctx, grad_output_beta1, grad_output_beta2):

        grad_alpha1 = grad_output_beta1
        grad_alpha2 = grad_output_beta2

        return grad_alpha1, grad_alpha2, None, None


if __name__ == "__main__":
    sr1 = torch.rand([20, 30], requires_grad=True, dtype=torch.float64)
    sr2 = torch.rand([20, 30], requires_grad=True, dtype=torch.float64)
    speed_accu = torch.rand(1, requires_grad=True, dtype=torch.float64)
    speed_curr = torch.rand(1, requires_grad=True, dtype=torch.float64)
    a1 = torch.rand(1, requires_grad=True, dtype=torch.float64)
    a2 = torch.rand(1, requires_grad=True, dtype=torch.float64)
    b1 = torch.rand(1, requires_grad=True, dtype=torch.float64)
    b2 = torch.rand(1, requires_grad=True, dtype=torch.float64)

    torch.autograd.gradcheck(ConditionFunction.apply, (a1, a2, b1, b2,), eps=1e-1)

    pass
