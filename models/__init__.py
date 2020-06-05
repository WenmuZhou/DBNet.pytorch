# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:55
# @Author  : zhoujun
from . import model
from .losses import DB_loss


def get_model(config):
    _model = getattr(model, config['type'])(config['args'])
    return _model


def get_loss(config):
    return getattr(DB_loss, config['type'])(**config['args'])
