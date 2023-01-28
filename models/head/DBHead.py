# -*- coding: utf-8 -*-
# @Time    : 2019/12/4 14:54
# @Author  : zhoujun
import torch
from torch import nn
import torch.nn.functional as F

class DBHead(nn.Module):
    def __init__(self, in_channels, out_channels, k = 50):  # debug ==> 256 2 k=50
        super().__init__()
        self.k = k
        self.binarize = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            # ConvTranspose2d (self, in_channels, out_channels, kernel_size, stride=1,
            #                  padding=0, output_padding=0, groups=1, bias=True,
            #                  dilation=1, padding_mode='zeros'):
            #nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2), # 上采样两倍
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            #nn.ConvTranspose2d(in_channels // 4, 1, 2, 2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels//4, 1, 3, padding=1), # 311 大小不变
            nn.Sigmoid())
        self.binarize.apply(self.weights_init)

        self.thresh = self._init_thresh(in_channels)
        self.thresh.apply(self.weights_init)

    def forward(self, x):
        shrink_maps = self.binarize(x)
        threshold_maps = self.thresh(x)
        if self.training:
            binary_maps = self.step_function(shrink_maps, threshold_maps)
            y = torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
        else:
            y = torch.cat((shrink_maps, threshold_maps), dim=1)
        return y

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels, serial=False, smooth=False, bias=False):
        in_channels = inner_channels # 256
        if serial:
            in_channels += 1

        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels // 4, smooth=smooth, bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self, in_channels, out_channels, smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=True))
            return nn.Sequential(module_list)
        else:
            #return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
            return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1))

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
