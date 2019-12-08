# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 15:54
# @Author  : zhoujun
import argparse
import os
import shutil

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from tools.predict import Pytorch_model
from utils import draw_bbox,cal_recall_precison_f1

torch.backends.cudnn.benchmark = True

def main(model_path, img_folder, save_path, gpu_id):
    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_img_folder = os.path.join(save_path, 'img')
    if not os.path.exists(save_img_folder):
        os.makedirs(save_img_folder)
    save_txt_folder = os.path.join(save_path, 'result')
    if not os.path.exists(save_txt_folder):
        os.makedirs(save_txt_folder)
    img_paths = [os.path.join(img_folder, x) for x in os.listdir(img_folder)]
    model = Pytorch_model(model_path, gpu_id=gpu_id)
    total_frame = 0.0
    total_time = 0.0
    for img_path in tqdm(img_paths):
        img_name = os.path.basename(img_path).split('.')[0]
        save_name = os.path.join(save_txt_folder, 'res_' + img_name + '.txt')
        _, boxes_list, t = model.predict(img_path)
        total_frame += 1
        total_time += t
        img = draw_bbox(img_path, boxes_list, color=(0, 0, 255))
        cv2.imwrite(os.path.join(save_img_folder, '{}.jpg'.format(img_name)), img)
        np.savetxt(save_name, boxes_list.reshape(-1, 8), delimiter=',', fmt='%d')
    print('fps:{}'.format(total_frame / total_time))
    return save_txt_folder


def init_args():
    parser = argparse.ArgumentParser(description='DBNet.pytorch')
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--img_folder', required=True, type=str)
    parser.add_argument('--gt_folder', required=True, type=str)
    parser.add_argument('--save_folder', required=True, type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import sys

    project = 'DBNet.pytorch'  # 工作项目根目录
    sys.path.append(os.getcwd().split(project)[0] + project)

    args = init_args()

    save_folder = main(args.model_path, args.img_folder, args.save_folder, gpu_id=0)
    result = cal_recall_precison_f1(gt_path=args.gt_folder, result_path=save_folder)
    print(result)
