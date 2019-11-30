# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:55
# @Author  : zhoujun
from . import model, loss


def get_model(config):
    return getattr(model, config['arch']['type'])(config['arch']['args'])


def get_loss(config):
    return getattr(loss, config['loss']['type'])(**config['loss']['args'])
