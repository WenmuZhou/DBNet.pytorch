# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:55
# @Author  : zhoujun
from .model import Model
from .losses import build_loss

__all__ = ['build_loss', 'build_model']
support_model = ['Model']


def build_model(model_name, **kwargs):
    assert model_name in support_model, f'all support model is {support_model}'
    model = eval(model_name)(kwargs)
    return model
