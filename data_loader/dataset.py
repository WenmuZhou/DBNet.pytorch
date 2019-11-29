# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:54
# @Author  : zhoujun
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from data_loader.data_utils import image_label
from utils import order_points_clockwise


class ImageDataset(Dataset):
    def __init__(self, data_list: list, input_size: int, img_channel: int, shrink_ratio: float, transform=None,
                 target_transform=None):
        self.data_list = self.load_data(data_list)
        self.input_size = input_size
        self.img_channel = img_channel
        self.transform = transform
        self.target_transform = target_transform
        self.shrink_ratio = shrink_ratio

    def __getitem__(self, index):
        img_path, text_polys, text_tags = self.data_list[index]
        im = cv2.imread(img_path, 1 if self.img_channel == 3 else 0)
        if self.img_channel == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        img, score_map, training_mask = image_label(im, text_polys, text_tags, self.input_size,
                                                    self.shrink_ratio)
        # img = draw_bbox(img,text_polys)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            score_map = self.target_transform(score_map)
            training_mask = self.target_transform(training_mask)
        return img, score_map, training_mask

    def load_data(self, data_list: list) -> list:
        t_data_list = []
        for img_path, label_path in data_list:
            bboxs, text_tags = self._get_annotation(label_path)
            if len(bboxs) > 0:
                t_data_list.append((img_path, bboxs, text_tags))
            else:
                print('there is no suit bbox in {}'.format(label_path))
        return t_data_list

    def _get_annotation(self, label_path: str) -> tuple:
        boxes = []
        text_tags = []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                try:
                    box = order_points_clockwise(np.array(list(map(float, params[:8]))).reshape(-1, 2))
                    if cv2.arcLength(box, True) > 0:
                        boxes.append(box)
                        label = params[8]
                        if label == '*' or label == '###':
                            text_tags.append(False)
                        else:
                            text_tags.append(True)
                except:
                    print('load label failed on {}'.format(label_path))
        return np.array(boxes, dtype=np.float32), np.array(text_tags, dtype=np.bool)

    def __len__(self):
        return len(self.data_list)


class Batch_Balanced_Dataset(object):
    def __init__(self, dataset_list: list, ratio_list: list, module_args: dict,
                 phase: str = 'train'):
        """
        对datasetlist里的dataset按照ratio_list里对应的比例组合，似的每个batch里的数据按按照比例采样的
        :param dataset_list: 数据集列表
        :param ratio_list: 比例列表
        :param module_args: dataloader的配置
        :param phase: 训练集还是验证集
        """
        assert sum(ratio_list) == 1 and len(dataset_list) == len(ratio_list)

        self.dataset_len = 0
        self.data_loader_list = []
        self.dataloader_iter_list = []
        all_batch_size = module_args['loader']['train_batch_size'] if phase == 'train' else module_args['loader'][
            'val_batch_size']
        for _dataset, batch_ratio_d in zip(dataset_list, ratio_list):
            _batch_size = max(round(all_batch_size * float(batch_ratio_d)), 1)

            _data_loader = DataLoader(dataset=_dataset,
                                      batch_size=_batch_size,
                                      shuffle=module_args['loader']['shuffle'],
                                      num_workers=module_args['loader']['num_workers'])

            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))
            self.dataset_len += len(_dataset)

    def __iter__(self):
        return self

    def __len__(self):
        return min([len(x) for x in self.data_loader_list])

    def __next__(self):
        balanced_batch_images = []
        balanced_batch_score_maps = []
        balanced_batch_training_masks = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, score_map, training_mask = next(data_loader_iter)
                balanced_batch_images.append(image)
                balanced_batch_score_maps.append(score_map)
                balanced_batch_training_masks.append(training_mask)
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, score_map, training_mask = next(self.dataloader_iter_list[i])
                balanced_batch_images.append(image)
                balanced_batch_score_maps.append(score_map)
                balanced_batch_training_masks.append(training_mask)
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)
        balanced_batch_score_maps = torch.cat(balanced_batch_score_maps, 0)
        balanced_batch_training_masks = torch.cat(balanced_batch_training_masks, 0)
        return balanced_batch_images, balanced_batch_score_maps, balanced_batch_training_masks


if __name__ == '__main__':
    import torch
    from utils.util import show_img
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from torchvision import transforms

    train_data = ImageDataset(
        data_list=[
            (r'E:/zj/dataset/icdar2015/train/img/img_15.jpg', 'E:/zj/dataset/icdar2015/train/gt/gt_img_15.txt')],
        input_size=640,
        img_channel=3,
        shrink_ratio=0.5,
        transform=transforms.ToTensor()
    )
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False, num_workers=0)

    pbar = tqdm(total=len(train_loader))
    for i, (img, shrink_label_map, threshold_label_map) in enumerate(train_loader):
        print(shrink_label_map.shape, shrink_label_map[0][0].max())
        print(img.shape)
        print(shrink_label_map[0][-1].sum())
        # pbar.update(1)
        show_img((img[0].to(torch.float)).numpy().transpose(1, 2, 0), color=True)
        show_img((shrink_label_map[0].to(torch.float)).numpy(), color=False)
        show_img((threshold_label_map[0].to(torch.float)).numpy(), color=False)
        plt.show()

    pbar.close()
