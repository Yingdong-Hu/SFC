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
* Other dependencies

Create an new conda environment
```
conda create -n sfc python=3.8 -y
conda activate sfc
```
Install [PyTorch](https://pytorch.org/)==1.8.0, [torchvision](https://pytorch.org/vision/stable/index.html)==0.9.0 following official instructions. For example:
```
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```
Clone this repo and install required packages:
```
git clone https://github.com/Alxead/SFC.git
pip install opencv-python scikit-image matplotlib wandb
```

### Dataset Preparation 
We use [YouTube-VOS](https://competitions.codalab.org/competitions/19544#participate-get-data) to pre-train fine-grained correspondence network.

Download raw image frames (`train_all_frames.zip`). Move `ytvos.csv` from `code/data/` to the directory of YouTube-VOS dataset.

The overall file structure should look like:
```
youtube-vos
├── train_all_frames
│   └── JPEGImages
└── ytvos.csv
```


## Pre-training Fine-grained Correspondence Network
To pre-train with a single 24GB NVIDIA 3090 GPU, run:
```
python train.py \
--data-path /path/to/youtube-vos \
--output-dir ../checkpoints \
--enable-wandb True
``` 
Training time is about 25 hours.


## Evaluation: Label Propagation
The label propagation algorithm is based on the implementation of [Contrastive Random Walk](https://github.com/ajabri/videowalk).

### DAVIS
To evaluate a model on the DAVIS task, clone [davis2017-evaluation](https://github.com/davisvideochallenge/davis2017-evaluation) repository.
```
git clone https://github.com/davisvideochallenge/davis2017-evaluation $HOME/davis2017-evaluation
```
Download [DAVIS2017](https://davischallenge.org/davis2017/code.html) dataset from the official website. Modify the paths provided in `code/eval/davis_vallist.txt`

### Pre-trained Model
Our fine-grained correspondence network and other baseline models can be downloaded as following:

After downloading an pre-trained model, place it  under `checkpoints/` folder. Please don't modify the file names of these checkpoints.
### Inference and Evaluation
To evaluate our SFC, run:

**Step 1:Video object segmentation**
```
python test.py --filelist /path/to/davis/vallist.txt \
--model-type scratch --resume ../pretrained.pth --save-path /save/path \
--topk 10 --videoLen 20 --radius 12  --temperature 0.05  --cropSize -1
```

**Step 2:Post-Process**

**Step 3:Compute metrics**


This should give:

Here you'll find the command-lines to evaluate some baseline models.

<details>
<summary>
MoCo
</summary>

```
python main.py --eval --model deit_base_distilled_patch16_224 --resume https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth
```

giving
```
* Acc@1 83.372 Acc@5 96.482 loss 0.685
```
</details>


## Acknowledgement

<!-- ## Citation

If you find our work useful in your research, please cite:
```latex

``` -->
