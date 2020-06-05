# -*- coding: utf-8 -*-
# @Time    : 2019/12/4 14:54
# @Author  : zhoujun
import torch
from torch import nn


class ConvHead(nn.Module):
    def __init__(self, in_channels, out_channels,**kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)