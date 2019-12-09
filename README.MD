# Real-time Scene Text Detection with Differentiable Binarization

**note**: some code is inherited from [MhLiao/DB](https://github.com/MhLiao/DB)

[中文解读](https://zhuanlan.zhihu.com/p/94677957)


![network](imgs/paper/db.png)

## Install
```bash
conda create --name DBNet.pytorch -y
conda activate DBNet.pytorch

conda install ipython pip

# python dependencies
pip install -r requirement.txt

# install PyTorch with cuda-10.1
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# clone repo
git clone https://github.com/WenmuZhou/DBNet.pytorch.git
cd DBNet.pytorch/

# build deformable convolution opertor
cd models/modules/dcn
python setup.py build_ext --inplace
```

## Requirements
* pytorch 1.1+
* torchvision 0.3+
* gcc 4.9+

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
1. config the `dataset['train']['dataset'['data_path']'`,`dataset['validate']['dataset'['data_path']`in [config/icdar2015_resnet18_fpn_DBhead_polyLR.yaml](cconfig/icdar2015_resnet18_fpn_DBhead_polyLR.yaml)
2. single gpu train
```bash
bash singlel_gpu_train.sh
```
3. Multi-gpu training
```bash
bash multi_gpu_train.sh
```
## Test

[eval.py](tools/eval.py) is used to test model on test dataset

1. config `model_path`, `img_folder`, `gt_folder`, `save_folder` in [eval.sh](eval.sh)
2. use following script to test
```bash
bash eval.sh
```

## Predict 
[predict.py](tools/predict.py) is used to inference on single image

1. config `model_path`, `img_path`, in [predict.py](tools/predict.py)
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


### todo
- [ ] mutil gpu training

### reference
1. https://arxiv.org/pdf/1911.08947.pdf
2. https://github.com/WenmuZhou/PANet.pytorch
3. https://github.com/MhLiao/DB

**If this repository helps you，please star it. Thanks.**