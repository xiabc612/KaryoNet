# KaryoNet

## Introduction

## Dataset

Due to privacy concerns, chromosome datasets for research purpose are available by request at: https://docs.google.com/forms/d/e/1FAIpQLScqgmljLn-OV2h9z-kUYjJK9uH0jq72VQneIDKUGLzXEuBciQ/viewform?usp=sf_link 

## Installation

Dependency: Python3; PyTorch

Download the following pretrained model and put them in model/pretrain folder:

https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth

https://download.pytorch.org/models/resnet50-19c8e357.pth

## Usage

For training KaryoNet for R-band chromosome, please running:
    
    python train_Rband.py --model resnet50_GFIM_DAM --model_name R_model_resnet50_GFIM_DAM

For training KaryoNet for G-band chromosome, please running:
    
    python train_Gband.py --model resnet50_MFIM_DAM --model_name G_model_resnet50_MFIM_DAM

For training the baseline ResNet50 model for R-band chromosome, please running:

    python train_Rband.py --model resnet50 --model_name R_model_resnet50

For training the baseline ResNet50 model for G-band chromosome, please running:

    python train_Gband.py --model resnet50 --model_name G_model_resnet50

## Citation

## Contact

For any question, please file an issue or contact

    ChaoXia: xiabc612@gmail.com
