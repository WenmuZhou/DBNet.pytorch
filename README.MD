# Real-time Scene Text Detection with Differentiable Binarization

![](imgs/paper/db.png)

## Requirements
* pytorch 1.1+
* torchvision 0.3+
* pyclipper
* opencv3

## Download

TBD

## Data Preparation

train: prepare a text in the following format, use '\t' as a separator
```bash
/path/to/img.jpg path/to/label.txt
...
```
val:
use a folder
```bash
img/ store img
gt/ store gt file
```

## Train
1. config the `train_data_path`,`val_data_path`in [config.json](config.json)
2. use following script to run
```sh
python3 train.py
```

## Test

[eval.py](eval.py) is used to test model on test dataset

1. config `model_path`, `img_path`, `gt_path`, `save_path` in [eval.py](eval.py)
2. use following script to test
```sh
python3 eval.py
```

## Predict 
[predict.py](predict.py) is used to inference on single image

1. config `model_path`, `img_path`, in [predict.py](predict.py)
2. use following script to predict
```sh
python3 predict.py
```

The project is still under development.

<h2 id="Performance">Performance</h2>

### [ICDAR 2015](http://rrc.cvc.uab.es/?ch=4)
only train on ICDAR2015 dataset

| Method                   | image size (short size) |learning rate | Precision (%) | Recall (%) | F-measure (%) | FPS |
|:--------------------------:|:-------:|:--------:|:--------:|:------------:|:---------------:|:-----:|
| DB-ResNet-18(paper)  | 736 |0.007 | 86.8 | 78.4 | 82.3 | 48 |
| TBD  |736 |1e-3| TBD | TBD | TBD | TBD |


### examples
TBD


### reference
1. https://arxiv.org/pdf/1911.08947.pdf
2. https://github.com/WenmuZhou/PANet.pytorch

**If this repository helps you，please star it. Thanks.**