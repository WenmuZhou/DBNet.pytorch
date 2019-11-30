# -*- coding: utf-8 -*-
# @Time    : 2019/11/27 17:53
# @Author  : zhoujun
import torch
from torch import nn


class DB(nn.Module):
    def __init__(self, k=50):
        super().__init__()
        self.k = k

    def forward(self, shrink_maps, threshold_maps):
        b_maps = self.k * (shrink_maps - threshold_maps)
        return torch.sigmoid(b_maps)
