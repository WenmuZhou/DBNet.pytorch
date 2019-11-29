# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:51
# @Author  : zhoujun

name = 'PAN'
arch = {
    "type": "PANModel",  # name of model architecture to train
    "args": {
        'backbone': 'resnet18',
        'fpem_repeat': 2,  # fpem模块重复的次数
        'pretrained': True,  # backbone 是否使用imagesnet的预训练模型
        'segmentation_head': 'FPN' #分割头，FPN or FPEM_FFM
    }
}


data_loader = {
    "type": "ImageDataset",  # selecting data loader
    "args": {
        'dataset': {
            'train_data_path': [['dataset1.txt1', 'dataset1.txt2'], ['dataset2.txt1', 'dataset2.txt2']],
            'train_data_ratio': [0.5, 0.5],
            'val_data_path': ['path/to/test/'],
            'input_size': 640,
            'img_channel': 3,
            'shrink_ratio': 0.5  # cv or PIL
        },
        'loader': {
            'validation_split': 0.1,
            'train_batch_size': 16,
            'val_batch_size': 4,
            'shuffle': True,
            'pin_memory': False,
            'num_workers': 6
        }
    }
}
loss = {
    "type": "PANLoss",  # name of model architecture to train
    "args": {
        'alpha': 0.5,
        'beta': 0.25,
        'delta_agg': 0.5,
        'delta_dis': 3,
        'ohem_ratio': 3
    }
}

optimizer = {
    "type": "Adam",
    "args": {
        "lr": 0.001,
        "weight_decay": 0,
        "amsgrad": True
    }
}

lr_scheduler = {
    "type": "StepLR",
    "args": {
        "step_size": 200,
        "gamma": 0.1
    }
}

resume = {
    'restart_training': True,
    'checkpoint': ''
}

trainer = {
    # random seed
    'seed': 2,
    'gpus': [0],
    'epochs': 600,
    'display_interval': 10,
    'show_images_interval': 50,
    'resume': resume,
    'output_dir': 'output',
    'tensorboard': True
}

config_dict = {}
config_dict['name'] = name
config_dict['data_loader'] = data_loader
config_dict['arch'] = arch
config_dict['loss'] = loss
config_dict['optimizer'] = optimizer
config_dict['lr_scheduler'] = lr_scheduler
config_dict['trainer'] = trainer

from utils import save_json

save_json(config_dict, '../config.json')
