# -*- coding: utf-8 -*-
# @Time    : 2020/6/5 11:36
# @Author  : zhoujun
__all__ = ['build_loss']

from .DB_loss import DBLoss

support_loss = ['DBLoss']


def build_loss(loss_name, **kwargs):
    assert loss_name in support_loss, f'all support loss is {support_loss}'
    criterion = eval(loss_name)(**kwargs)
    return criterion
