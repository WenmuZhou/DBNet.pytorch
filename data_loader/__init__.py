# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:52
# @Author  : zhoujun

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import copy

from utils import get_datalist
from . import dataset


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


def train_val_split(ds, validation_split):
    '''

    :param ds: dataset
    :return:
    '''
    try:
        split = float(validation_split)
    except:
        raise RuntimeError('Train and val splitting ratio is invalid.')

    val_len = int(split * len(ds))
    train_len = len(ds) - val_len
    train, val = random_split(ds, [train_len, val_len])
    return train, val


def get_dataloader(module_name, module_args):
    train_transfroms = transforms.Compose([
        transforms.ColorJitter(brightness=0.5),
        transforms.ToTensor()
    ])

    # 创建数据集
    dataset_args = copy.deepcopy(module_args['dataset'])
    train_data_path = dataset_args.pop('train_data_path')
    dataset_args.pop('val_data_path')
    if module_name == 'ICDAR2015Dataset':
        train_data_list = get_datalist(train_data_path)
    elif module_name == 'SynthTextDataset':
        train_data_list = train_data_path
    else:
        raise NotImplementedError

    train_dataset = get_dataset(data_list=train_data_list, module_name=module_name, transform=train_transfroms, dataset_args=dataset_args)
    train_loader = DataLoader(dataset=train_dataset, batch_size=module_args['loader']['train_batch_size'],
                              shuffle=module_args['loader']['shuffle'], num_workers=module_args['loader']['num_workers'])
    return train_loader
