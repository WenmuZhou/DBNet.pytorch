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
        self.bce = nn.BCELoss(reduction=reduction)
        self.l1 = nn.L1Loss(reduction=reduction)
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction

    def forward(self, outputs, gt_shrink_labels, gt_threshold_labels):
        shrink_maps = outputs[:, 0, :, :]
        threshold_maps = outputs[:, 1, :, :]
        binary_maps = outputs[:, 2, :, :]

        # 计算 text loss
        selected_masks = self.ohem_batch(shrink_maps, gt_shrink_labels)
        selected_masks = selected_masks.to(outputs.device)

        loss_shrink_map = self.bce_loss(shrink_maps, gt_shrink_labels, selected_masks)
        loss_binary_map = self.bce_loss(binary_maps, gt_shrink_labels, selected_masks)
        loss_threshold_map = self.threshold_loss(threshold_maps, gt_threshold_labels, gt_shrink_labels)

        loss_all = loss_shrink_map + self.alpha * loss_binary_map + self.beta * loss_threshold_map
        return loss_all, loss_shrink_map, loss_binary_map, loss_threshold_map

    def threshold_loss(self, threshold_maps, gt_threshold_maps, gt_shrink_labels):
        selected_masks = (gt_threshold_maps > 0) | (gt_shrink_labels > 0)
        if selected_masks.sum() == 0:
            return torch.tensor(0.0, device=threshold_maps.device, requires_grad=True)
        threshold_maps = torch.sigmoid(threshold_maps)
        threshold_maps = threshold_maps[selected_masks]
        gt_threshold_maps = gt_threshold_maps[selected_masks]
        loss = self.l1(threshold_maps, gt_threshold_maps)
        return loss

    def bce_loss(self, input, target, mask):
        if mask.sum() == 0:
            return torch.tensor(0.0, device=input.device, requires_grad=True)
        input = torch.sigmoid(input)
        target[target <= 0.5] = 0
        target[target > 0.5] = 1
        input = input[mask.bool()]
        target = target[mask.bool()]
        loss = self.bce(input, target)
        return loss

    def ohem_single(self, score, gt_text):
        pos_num = (int)(np.sum(gt_text > 0.5))

        if pos_num == 0:
            selected_mask = np.zeros_like(score)
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
            return selected_mask

        neg_num = (int)(np.sum(gt_text <= 0.5))
        neg_num = (int)(min(pos_num * self.ohem_ratio, neg_num))

        if neg_num == 0:
            selected_mask = np.zeros_like(score)
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
            return selected_mask

        neg_score = score[gt_text <= 0.5]
        neg_score_sorted = np.sort(-neg_score)
        threshold = -neg_score_sorted[neg_num - 1]
        selected_mask = (score >= threshold) | (gt_text > 0.5)
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    def ohem_batch(self, scores, gt_texts):
        scores = scores.data.cpu().numpy()
        gt_texts = gt_texts.data.cpu().numpy()
        selected_masks = []
        for i in range(scores.shape[0]):
            selected_masks.append(self.ohem_single(scores[i, :, :], gt_texts[i, :, :]))

        selected_masks = np.concatenate(selected_masks, 0)
        selected_masks = torch.from_numpy(selected_masks).float()

        return selected_masks
