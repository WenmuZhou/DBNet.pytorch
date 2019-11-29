# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:52
# @Author  : zhoujun

from torch.utils.data import DataLoader
from torchvision import transforms
import copy
import pathlib
from . import dataset


def get_datalist(train_data_path, validation_split=0.1):
    """
    获取训练和验证的数据list
    :param train_data_path: 训练的dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :param validation_split: 验证集的比例，当val_data_path为空时使用
    :return:
    """
    train_data_list = []
    for train_path in train_data_path:
        train_data = []
        for p in train_path:
            with open(p, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
                    if len(line) > 1:
                        img_path = pathlib.Path(line[0].strip(' '))
                        label_path = pathlib.Path(line[1].strip(' '))
                        if img_path.exists() and img_path.stat().st_size > 0 and label_path.exists() and label_path.stat().st_size > 0:
                            train_data.append((str(img_path), str(label_path)))
        train_data_list.append(train_data)
    return train_data_list


def get_dataset(data_list, module_name, transform, dataset_args):
    """
    获取训练dataset
    :param data_list: dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :param module_name: 所使用的自定义dataset名称，目前只支持data_loaders.ImageDataset
    :param transform: 该数据集使用的transforms
    :param dataset_args: module_name的参数
    :return: 如果data_path列表不为空，返回对于的ConcatDataset对象，否则None
    """
    s_dataset = getattr(dataset, module_name)(transform=transform, data_list=data_list,
                                              **dataset_args)
    return s_dataset


def get_dataloader(module_name, module_args):
    train_transfroms = transforms.Compose([
        transforms.ColorJitter(brightness=0.5),
        transforms.ToTensor()
    ])

    # 创建数据集
    dataset_args = copy.deepcopy(module_args['dataset'])
    train_data_path = dataset_args.pop('train_data_path')
    train_data_ratio = dataset_args.pop('train_data_ratio')
    dataset_args.pop('val_data_path')
    train_data_list = get_datalist(train_data_path, module_args['loader']['validation_split'])
    train_dataset_list = []
    for train_data in train_data_list:
        train_dataset_list.append(get_dataset(data_list=train_data,
                                              module_name=module_name,
                                              transform=train_transfroms,
                                              dataset_args=dataset_args))

    if len(train_dataset_list) > 1:
        train_loader = dataset.Batch_Balanced_Dataset(dataset_list=train_dataset_list,
                                                      ratio_list=train_data_ratio,
                                                      module_args=module_args,
                                                      phase='train')
    elif len(train_dataset_list) == 1:
        train_loader = DataLoader(dataset=train_dataset_list[0],
                                  batch_size=module_args['loader']['train_batch_size'],
                                  shuffle=module_args['loader']['shuffle'],
                                  num_workers=module_args['loader']['num_workers'])
        train_loader.dataset_len = len(train_dataset_list[0])
    else:
        raise Exception('no images found')
    return train_loader
