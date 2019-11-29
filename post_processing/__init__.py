# -*- coding: utf-8 -*-
# @Time    : 2019/9/8 14:18
# @Author  : zhoujun
import os
import cv2
import torch
import subprocess
import numpy as np
import pyclipper

BASE_DIR = os.path.dirname(os.path.realpath(__file__))



def de_shrink(poly, r = 0.5):
    d_i = cv2.contourArea(poly) * r / cv2.arcLength(poly, True)
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    shrinked_poly = np.array(pco.Execute(d_i))
    return shrinked_poly


def decode(preds, threshold=0.2, min_area=5):
    """
    在输出上使用sigmoid 将值转换为置信度，并使用阈值来进行文字和背景的区分
    :param preds: 网络输出
    :param scale: 网络的scale
    :param threshold: sigmoid的阈值
    :return: 最后的输出图和文本框
    """

    if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
        raise RuntimeError('Cannot compile pse: {}'.format(BASE_DIR))
    from .pse import get_points, get_num
    shrink_map = preds[0, :, :].detach().cpu().numpy()
    score_map = shrink_map.astype(np.float32)
    shrink_map = shrink_map > threshold

    label_num, label = cv2.connectedComponents(shrink_map.astype(np.uint8), connectivity=4)

    bbox_list = []
    label_points = get_points(label, score_map, label_num)
    for label_value, label_point in label_points.items():
        score_i = label_point[0]
        label_point = label_point[2:]
        points = np.array(label_point, dtype=int).reshape(-1, 2)

        if points.shape[0] < min_area:
            continue

        # if score_i < 0.93:
        #     continue

        rect = cv2.minAreaRect(points)
        poly = cv2.boxPoints(rect)
        shrinked_poly = de_shrink(poly)
        if shrinked_poly.size == 0:
            continue
        rect = cv2.minAreaRect(shrinked_poly)
        shrinked_poly = cv2.boxPoints(rect).astype(int)
        if cv2.contourArea(shrinked_poly) < 100:
            continue
        bbox_list.append([shrinked_poly[1], shrinked_poly[2], shrinked_poly[3], shrinked_poly[0]])
    return label, np.array(bbox_list)


def decode_py(preds, threshold=0.7311, min_area=5):
    shrink_map = preds[0, :, :].detach().cpu().numpy()
    # score_map = shrink_map.astype(np.float32)
    shrink_map = shrink_map > threshold

    label_num, label = cv2.connectedComponents(shrink_map.astype(np.uint8), connectivity=4)
    bbox_list = []
    for label_idx in range(1, label_num):
        points = np.array(np.where(label == label_idx)).transpose((1, 0))[:, ::-1]
        if points.shape[0] < min_area:
            continue

        # score_i = np.mean(score_map[label == label_idx])
        # if score_i < 0.93:
        #     continue

        rect = cv2.minAreaRect(points)
        poly = cv2.boxPoints(rect).astype(int)

        shrinked_poly = de_shrink(poly)
        if shrinked_poly.size == 0:
            continue
        rect = cv2.minAreaRect(shrinked_poly)
        shrinked_poly = cv2.boxPoints(rect).astype(int)
        if cv2.contourArea(shrinked_poly) < 100:
            continue

        bbox_list.append([shrinked_poly[1], shrinked_poly[2], shrinked_poly[3], shrinked_poly[0]])
    return label, np.array(bbox_list)
