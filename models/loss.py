# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:56
# @Author  : zhoujun
import itertools
import torch
from torch import nn
import numpy as np


class DBLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=10, ohem_ratio=3, reduction='mean'):
        """
        Implement PSE Loss.
        :param alpha: binary_map loss 前面的系数
        :param beta: threshold_map loss 前面的系数
        :param ohem_ratio: OHEM的比例
        :param reduction: 'mean' or 'sum'对 batch里的loss 算均值或求和
        """
        super().__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCELoss()
        self.l1 = nn.L1Loss()
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction

    def forward(self, outputs, gt_shrink_labels, gt_threshold_labels):
        shrink_maps = outputs[:, 0, :, :]
        threshold_maps = outputs[:, 1, :, :]
        binary_maps = outputs[:, 2, :, :]
        loss_shrink_maps = []
        loss_binary_maps = []
        loss_threshold_maps = []
        for shrink_map, threshold_map, binary_map, gt_shrink_label, gt_threshold_label in zip(shrink_maps, threshold_maps, binary_maps, gt_shrink_labels,
                                                                                              gt_threshold_labels):
            loss_shrink_map = self.bce_loss(shrink_map, gt_shrink_label)
            loss_binary_map = self.bce_loss(torch.sigmoid(binary_map), gt_shrink_label) # 理论上binary_map已经sigmoid了，不知道为什么这里加了sigmoid才会收敛
            loss_threshold_map = self.threshold_loss(threshold_map, gt_threshold_label, gt_shrink_label)
            loss_shrink_maps.append(loss_shrink_map)
            loss_binary_maps.append(loss_binary_map)
            loss_threshold_maps.append(loss_threshold_map)
        loss_shrink_maps = torch.stack(loss_shrink_maps)
        loss_binary_maps = torch.stack(loss_binary_maps)
        loss_threshold_maps = torch.stack(loss_threshold_maps)
        # mean or sum
        if self.reduction == 'mean':
            loss_shrink_maps = loss_shrink_maps.mean()
            loss_binary_maps = loss_binary_maps.mean()
            loss_threshold_maps = loss_threshold_maps.mean()
        elif self.reduction == 'sum':
            loss_shrink_maps = loss_shrink_maps.sum()
            loss_binary_maps = loss_binary_maps.sum()
            loss_threshold_maps = loss_threshold_maps.sum()
        else:
            raise NotImplementedError
        loss_all = loss_shrink_maps + self.alpha * loss_binary_maps + self.beta * loss_threshold_maps
        return loss_all, loss_shrink_maps, loss_binary_maps, loss_threshold_maps

    def threshold_loss(self, threshold_maps, gt_threshold_maps, gt_shrink_labels):
        selected_masks = (gt_threshold_maps > 0) | (gt_shrink_labels > 0)
        if selected_masks.sum() == 0:
            return torch.tensor(0.0, device=threshold_maps.device, requires_grad=True)
        threshold_maps = threshold_maps[selected_masks]
        gt_threshold_maps = gt_threshold_maps[selected_masks]
        loss = self.l1(threshold_maps, gt_threshold_maps)
        return loss

    def bce_loss(self, input, target):
        # ohem
        mask = self.ohem_single(input.data.cpu().numpy(), target.data.cpu().numpy())
        mask = torch.from_numpy(mask).float()
        mask = mask.to(input.device)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=input.device, requires_grad=True)
        input = input[mask.bool()]
        target = target[mask.bool()]
        loss = self.bce(input, target)
        return loss

    def ohem_single(self, score, gt_text):
        pos_num = (int)(np.sum(gt_text > 0.5))

        if pos_num == 0:
            selected_mask = np.ones_like(score)
            return selected_mask.astype('float32')

        neg_num = (int)(np.sum(gt_text <= 0.5))
        neg_num = (int)(min(pos_num * self.ohem_ratio, neg_num))

        if neg_num == 0:
            selected_mask = np.ones_like(score)
            return selected_mask.astype('float32')

        neg_score = score[gt_text <= 0.5]
        neg_score_sorted = np.sort(-neg_score)
        threshold = -neg_score_sorted[neg_num - 1]
        selected_mask = (score >= threshold) | (gt_text > 0.5)
        return selected_mask.astype('float32')