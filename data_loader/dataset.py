# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:54
# @Author  : zhoujun
import copy
import pathlib

import cv2
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import DataLoader

from base import BaseDataSet
from utils import order_points_clockwise


class ICDAR2015Dataset(BaseDataSet):
    def __init__(self, data_path: list, img_model, pre_processes, filter_keys, transform=None, **kwargs):
        super().__init__(data_path, img_model, pre_processes, transform)
        self.filter_keys = filter_keys

    def __getitem__(self, index):
        data = self.data_list[index]
        im = cv2.imread(data['img_path'], 1 if self.img_model != 'GRAY' else 0)
        if self.img_model == 'RGB':
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        data['img'] = im
        data['shape'] = [im.shape[0], im.shape[1]]
        data = self.allpy_pre_processes(copy.deepcopy(data))
        # img, shrink_label_map, threshold_label_map = image_label(im, gt, self.input_size, self.shrink_ratio, self.modules)
        # img = draw_bbox(img,text_polys)
        if self.transform:
            data['img'] = self.transform(data['img'])
        data['text_polys'] = data['text_polys'].tolist()
        if len(self.filter_keys):
            data_dict = {}
            for k, v in data.items():
                if k not in self.filter_keys:
                    data_dict[k] = v
        else:
            data_dict = data
        return data_dict

    def load_data(self, data_list: list) -> list:
        t_data_list = []
        for img_path, label_path in data_list:
            data = self._get_annotation(label_path)
            if len(data['text_polys']) > 0:
                item = {'img_path': img_path, 'img_name': pathlib.Path(img_path).stem}
                item.update(data)
                t_data_list.append(item)
            else:
                print('there is no suit bbox in {}'.format(label_path))
        return t_data_list

    def _get_annotation(self, label_path: str) -> dict:
        boxes = []
        texts = []
        ignores = []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                try:
                    box = order_points_clockwise(np.array(list(map(float, params[:8]))).reshape(-1, 2))
                    if cv2.contourArea(box) > 0:
                        boxes.append(box)
                        label = params[8]
                        texts.append(label)
                        ignores.append(label in ['*', '###'])
                except:
                    print('load label failed on {}'.format(label_path))
        data = {
            'text_polys': np.array(boxes),
            'texts': texts,
            'ignore_tags': ignores,
        }
        return data

    def __len__(self):
        return len(self.data_list)


class SynthTextDataset(BaseDataSet):

    def __init__(self, data_list: str, input_size: int, img_channel: int, shrink_ratio: float, transform=None,
                 target_transform=None):
        self.input_size = input_size
        self.img_channel = img_channel
        self.transform = transform
        self.target_transform = target_transform
        self.shrink_ratio = shrink_ratio
        self.dataRoot = pathlib.Path(data_list)
        if not self.dataRoot.exists():
            raise FileNotFoundError('Dataset folder is not exist.')

        self.targetFilePath = self.dataRoot / 'gt.mat'
        if not self.targetFilePath.exists():
            raise FileExistsError('Target file is not exist.')
        targets = {}
        sio.loadmat(self.targetFilePath, targets, squeeze_me=True, struct_as_record=False,
                    variable_names=['imnames', 'wordBB', 'txt'])

        self.imageNames = targets['imnames']
        self.wordBBoxes = targets['wordBB']
        self.transcripts = targets['txt']

    def __getitem__(self, index):
        """

        :param index:
        :return:
            imageName: path of image
            wordBBox: bounding boxes of words in the image
            transcript: corresponding transcripts of bounded words
        """
        imageName = self.imageNames[index]
        wordBBoxes = self.wordBBoxes[index]  # 2 * 4 * num_words
        transcripts = self.transcripts[index]
        im = cv2.imread((self.dataRoot / imageName).as_posix())
        imagePath = pathlib.Path(imageName)
        wordBBoxes = np.expand_dims(wordBBoxes, axis=2) if (wordBBoxes.ndim == 2) else wordBBoxes
        _, _, numOfWords = wordBBoxes.shape
        text_polys = wordBBoxes.reshape([8, numOfWords], order='F').T  # num_words * 8
        text_polys = text_polys.reshape(numOfWords, 4, 2)  # num_of_words * 4 * 2
        transcripts = [word for line in transcripts for word in line.split()]

        img, shrink_label_map, threshold_label_map = image_label(im, text_polys, transcripts, self.input_size,
                                                                 self.shrink_ratio)
        # img = draw_bbox(img,text_polys)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            shrink_label_map = self.target_transform(shrink_label_map)
            threshold_label_map = self.target_transform(threshold_label_map)
        return img, shrink_label_map, threshold_label_map

    def __len__(self):
        return len(self.imageNames)


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
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from torchvision import transforms
    import anyconfig
    from utils import get_datalist, parse_config
    from utils.util import show_img

    import PIL


    class ICDARCollectFN():
        def __init__(self, *args, **kwargs):
            self.to_tensor = transforms.ToTensor()

        def __call__(self, batch):
            data_dict = {}
            to_tensor_keys = []
            for sample in batch:
                for k, v in sample.items():
                    if k not in data_dict:
                        data_dict[k] = []
                    if isinstance(v, (np.ndarray, torch.Tensor, PIL.Image.Image)):
                        v = self.to_tensor(v)
                        if k not in to_tensor_keys:
                            to_tensor_keys.append(k)
                    data_dict[k].append(v)
            for k in to_tensor_keys:
                data_dict[k] = torch.stack(data_dict[k], 0)
            return data_dict


    config = anyconfig.load('E:\zj\code\DBNet.pytorch\config\icdar2015_resnet18_fpn_DBhead_polyLR.yaml')
    config = parse_config(config)
    dataset_args = config['dataset']['train']['dataset']['args']
    # dataset_args.pop('data_path')
    # data_list = [(r'E:/zj/dataset/icdar2015/train/img/img_15.jpg', 'E:/zj/dataset/icdar2015/train/gt/gt_img_15.txt')]
    data_list = get_datalist(dataset_args.pop('data_path'))
    train_data = ICDAR2015Dataset(data_path=data_list, transform=None, **dataset_args)
    icdar_collate_fn = ICDARCollectFN()
    train_loader = DataLoader(dataset=train_data, batch_size=2, shuffle=False, collate_fn=icdar_collate_fn, num_workers=0)

    pbar = tqdm(total=len(train_loader))
    for i, data in enumerate(train_loader):
        # img = data['img']
        # shrink_label = data['shrink_map']
        # threshold_label = data['threshold_map']
        # print(threshold_label.shape, threshold_label.shape, img.shape)
        pbar.update(1)
        # show_img((img[0].to(torch.float)).numpy().transpose(1, 2, 0), title='img', color=True)
        # show_img((shrink_label[0].to(torch.float)).numpy(), title='shrink_label', color=False)
        # show_img((threshold_label[0].to(torch.float)).numpy(), title='threshold_label', color=False)
        # show_img(((threshold_label + shrink_label)[0].to(torch.float)).numpy(), title='threshold_label+threshold_label', color=False)
        # plt.show()

    pbar.close()
