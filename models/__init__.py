# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:55
# @Author  : zhoujun
from . import model, loss


def get_model(config, distributed=False, local_rank=0):
    _model = getattr(model, config['type'])(config['args'])
    if distributed:
        import torch
        _model = torch.nn.parallel.DistributedDataParallel(_model, device_ids=[local_rank], output_device=local_rank)
    return _model


def get_loss(config):
    return getattr(loss, config['type'])(**config['args'])
