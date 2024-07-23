# KaryoNet

## Introduction

A chromosome recognition network with end-to-end differentiable combinatorial optimization.

## Dataset

Due to privacy concerns, chromosome datasets for research purpose are available by request at: https://docs.google.com/forms/d/e/1FAIpQLScqgmljLn-OV2h9z-kUYjJK9uH0jq72VQneIDKUGLzXEuBciQ/viewform?usp=sf_link 

## Installation

Dependency: Python3; PyTorch

Download the following pretrained model and put them in model/pretrain folder:

https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth

https://download.pytorch.org/models/resnet50-19c8e357.pth

## Usage

For training KaryoNet for R-band chromosome, please running:
    
    python train_Rband.py --model resnet50_MFIM_DAM --model_name R_model_resnet50_MFIM_DAM

For training KaryoNet for G-band chromosome, please running:
    
    python train_Gband.py --model resnet50_MFIM_DAM --model_name G_model_resnet50_MFIM_DAM

For training the baseline ResNet50 model for R-band chromosome, please running:

    python train_Rband.py --model resnet50 --model_name R_model_resnet50

For training the baseline ResNet50 model for G-band chromosome, please running:

    python train_Gband.py --model resnet50 --model_name G_model_resnet50

## Citing KaryoNet

If you find KaryoNet useful in your research, please consider citing:

    @article{xia2023karyonet,  
      title={KaryoNet: Chromosome Recognition with End-to-End Combinatorial Optimization Network},  
      author={Xia, Chao and Wang, Jiyue and Qin, Yulei and Wen, Juan and Liu, Zhaojiang and Song, Ning and Wu, Lingqian and Chen, Bing and Gu, Yun and Yang, Jie},  
      journal={IEEE Transactions on Medical Imaging},  
      year={2023},  
      publisher={IEEE}  
    }

## Contact

For any question, please file an issue or contact

    ChaoXia: xiabc612@gmail.com
