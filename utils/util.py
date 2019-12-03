# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:59
# @Author  : zhoujun
import time
import json
import cv2
import pathlib
import numpy as np
import matplotlib.pyplot as plt


def setup_logger(log_file_path: str = None):
    import logging
    logging._warn_preinit_stderr = 0
    logger = logging.getLogger('DBNet.pytorch')
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file_path is not None:
        file_handle = logging.FileHandler(log_file_path)
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    logger.info('logger init finished')
    return logger


# --exeTime
def exe_time(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        print("{} cost {:.3f}s".format(func.__name__, time.time() - t0))
        return back

    return newFunc


def save_json(data, json_path):
    with open(json_path, mode='w', encoding='utf8') as f:
        json.dump(data, f, indent=4)


def load_json(json_path):
    with open(json_path, mode='r', encoding='utf8') as f:
        data = json.load(f)
    return data


def show_img(imgs: np.ndarray, color=False):
    if (len(imgs.shape) == 3 and color) or (len(imgs.shape) == 2 and not color):
        imgs = np.expand_dims(imgs, axis=0)
    for img in imgs:
        plt.figure()
        plt.imshow(img, cmap=None if color else 'gray')


def draw_bbox(img_path, result, color=(255, 0, 0), thickness=2):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)
        cv2.line(img_path, tuple(point[0]), tuple(point[1]), color, thickness)
        cv2.line(img_path, tuple(point[1]), tuple(point[2]), color, thickness)
        cv2.line(img_path, tuple(point[2]), tuple(point[3]), color, thickness)
        cv2.line(img_path, tuple(point[3]), tuple(point[0]), color, thickness)
    return img_path


def cal_text_score(texts, gt_texts, running_metric_text, thred=0.5):
    pred_text = texts.data.cpu().numpy()
    pred_text[pred_text <= thred] = 0
    pred_text[pred_text > thred] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.data.cpu().numpy()
    gt_text = gt_text.astype(np.int32)
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text


def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def order_points_clockwise_list(pts):
    pts = pts.tolist()
    pts.sort(key=lambda x: (x[1], x[0]))
    pts[:2] = sorted(pts[:2], key=lambda x: x[0])
    pts[2:] = sorted(pts[2:], key=lambda x: -x[0])
    pts = np.array(pts)
    return pts

def get_datalist(train_data_path):
    """
    获取训练和验证的数据list
    :param train_data_path: 训练的dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :return:
    """
    train_data = []
    for p in train_data_path:
        with open(p, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
                if len(line) > 1:
                    img_path = pathlib.Path(line[0].strip(' '))
                    label_path = pathlib.Path(line[1].strip(' '))
                    if img_path.exists() and img_path.stat().st_size > 0 and label_path.exists() and label_path.stat().st_size > 0:
                        train_data.append((str(img_path), str(label_path)))
    return train_data

if __name__ == '__main__':
    log = setup_logger('1')
    log.info('1')
