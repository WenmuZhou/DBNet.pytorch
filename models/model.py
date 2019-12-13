# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:57
# @Author  : zhoujun

import torch.nn.functional as F

from models.modules import *

backbone_dict = {
    'resnet18': {'models': resnet18, 'out': [64, 128, 256, 512]},
    'deformable_resnet18': {'models': deformable_resnet18, 'out': [64, 128, 256, 512]},
    'resnet34': {'models': resnet34, 'out': [64, 128, 256, 512]},
    'resnet50': {'models': resnet50, 'out': [256, 512, 1024, 2048]},
    'deformable_resnet50': {'models': deformable_resnet50, 'out': [256, 512, 1024, 2048]},
    'resnet101': {'models': resnet101, 'out': [256, 512, 1024, 2048]},
    'resnet152': {'models': resnet152, 'out': [256, 512, 1024, 2048]},
    'shufflenetv2': {'models': shufflenet_v2_x1_0, 'out': [24, 116, 232, 464]}
}

segmentation_body_dict = {'FPN': FPN, 'FPEM_FFM': FPEM_FFM}
segmentation_head_dict = {'conv': ConvHead, 'DBHead': DBHead}


# 'MobileNetV3_Large': {'models': MobileNetV3_Large, 'out': [24, 40, 160, 160]},
# 'MobileNetV3_Small': {'models': MobileNetV3_Small, 'out': [16, 24, 48, 96]},
# 'shufflenetv2': {'models': shufflenet_v2_x1_0, 'out': [24, 116, 232, 464]}}


class DBModel(nn.Module):
    def __init__(self, model_config: dict):
        """
        PANnet
        :param model_config: 模型配置
        """
        super().__init__()
        backbone = model_config['backbone']
        pretrained = model_config['pretrained']
        segmentation_body = model_config['segmentation_body']['type']
        segmentation_head = model_config['segmentation_head']['type']

        assert backbone in backbone_dict, 'backbone must in: {}'.format(backbone_dict)
        assert segmentation_body in segmentation_body, 'segmentation_head must in: {}'.format(segmentation_body)
        assert segmentation_head in segmentation_head_dict, 'segmentation_head must in: {}'.format(segmentation_head_dict)

        backbone_model, backbone_out = backbone_dict[backbone]['models'], backbone_dict[backbone]['out']
        self.backbone = backbone_model(pretrained=pretrained)
        self.segmentation_body = segmentation_body_dict[segmentation_body](backbone_out, **model_config['segmentation_body']['args'])
        self.segmentation_head = segmentation_head_dict[segmentation_head](self.segmentation_body.out_channels, **model_config['segmentation_head']['args'])
        self.name = '{}_{}_{}'.format(backbone, segmentation_body, segmentation_head)

    def forward(self, x):
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        segmentation_body_out = self.segmentation_body(backbone_out)
        y = self.segmentation_head(segmentation_body_out)
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
        return y


if __name__ == '__main__':
    device = torch.device('cuda:0')
    x = torch.zeros(1, 3, 640, 640).to(device)

    model_config = {
        'backbone': 'shufflenetv2',
        'fpem_repeat': 2,  # fpem模块重复的次数
        'pretrained': True,  # backbone 是否使用imagesnet的预训练模型
        'out_channels': 2,
        "k": 50,
        'segmentation_head': 'FPN'  # 分割头，FPN or FPEM_FFM
    }
    model = DBModel(model_config=model_config).to(device)
    import time

    tic = time.time()
    y = model(x)
    print(time.time() - tic)
    print(y.shape)
    # print(model)
    # torch.save(model.state_dict(), 'PAN.pth')
