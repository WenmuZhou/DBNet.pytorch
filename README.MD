# Real-time Scene Text Detection with Differentiable Binarization

**note**: 原始版本 [DBNet.pytorch](https://github.com/WenmuZhou/DBNet.pytorch)

[中文解读](https://zhuanlan.zhihu.com/p/94677957)

![network](imgs/paper/db.jpg)

## 安装环境

请参考原始版本的[Readme](https://github.com/WenmuZhou/DBNet.pytorch/blob/master/README.MD)



## 修改之处

本repo为了可以使用tensorRT加速，将反卷积操作全部改为upsample。比如

```python
 # 原始版本
nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2), # 上采样两倍
# 修改版本 
nn.Upsample(scale_factor=2, mode='nearest'),
```

更多的修改，请看代码：

```
models/head/DBHead.py 
models/model.py 
models/neck/FPN.py
```

## 模型

修改后代码训练的模型地址：[渣云：访问密码 myj4 ](https://pan.baidu.com/s/10Ff-0AJkkpC9jGWdNSsN6g)

目前没有训练完成，相比原版模型（1200epoch），只训练了500epoch。精度：90.0  召回率：68.2。 

可以自己去训练。

## TensorRT版本

[地址]: https://github.com/BaofengZan/DBNet-TensorRT

 