# -*- coding: utf-8 -*-
# @Time    : 2019/12/4 13:12
# @Author  : zhoujun
from torch.utils.data import Dataset
from data_loader.modules import *


class BaseDataSet(Dataset):

    def __init__(self, data_path: list, img_mode, pre_processes, transform=None,
                 target_transform=None):
        assert img_mode in ['RGB', 'BRG', 'GRAY']
        self.data_list = self.load_data(data_path)
        self.img_mode = img_mode
        self.transform = transform
        self.target_transform = target_transform
        self._init_apre_processes(pre_processes)

    def _init_apre_processes(self, pre_processes):
        self.aug = []
        if pre_processes is not None:
            for aug in pre_processes:
                if 'args' not in aug:
                    args = {}
                else:
                    args = aug['args']
                if isinstance(args, dict):
                    cls = eval(aug['type'])(**args)
                else:
                    cls = eval(aug['type'])(args)
                self.aug.append(cls)

    def load_data(self, data_list: list) -> list:
        raise NotImplementedError

    def allpy_pre_processes(self, data):
        for aug in self.aug:
            data = aug(data)
        return data

    def __len__(self):
        return len(self.data_list)
