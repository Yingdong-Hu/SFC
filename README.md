# Semantic-Aware Fine-Grained Correspondence

This repository is the official PyTorch implementation for SFC introduced in the paper:

Semantic-Aware Fine-Grained Correspondence. ECCV 2022 (**Oral**)
<br>
Yingdong Hu, Renhao Wang, Kaifeng Zhang, and Yang Gao
<br>



## Installation

### Dependency Setup
* Python 3.8
* PyTorch 1.8.0
* davis2017-evaluation

Create an new conda environment
```
conda create -n sfc python=3.8 -y
conda activate sfc
```
Install [PyTorch](https://pytorch.org/)==1.8.0, [torchvision](https://pytorch.org/vision/stable/index.html)==0.9.0 following official instructions. For example:
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
Clone this repo and install required packages:
```
git clone https://github.com/facebookresearch/ConvNeXt
pip install timm==0.3.2 tensorboardX six
```

### Dataset Preparation 
Youtube-VOS and DAVIS-2017

## Pre-training Fine-grained Correspondence Network
To pre-train with a single 24GB NVIDIA 3090 GPU, run:
```
python train.py \
--data-path /mnt/huyingdong/youtube-vos \
--output-dir ../checkpoints \
--enable-wandb True
``` 
Training time is about 25 hours.


## Evaluation: Label Propagation

## Acknowledgement

<!-- ## Citation

If you find our work useful in your research, please cite:
```latex

``` -->
