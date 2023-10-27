
<div align="center" markdown>

<img src="https://user-images.githubusercontent.com/48245050/182847473-9a35f213-c27b-4abd-bd64-c73bf80fb056.jpg"/>  

# Train MMSegmentation

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Demo">Demo</a> •
  <a href="#How-To-Use-Your-Trained-Model-Outside-Supervisely">How To Use Your Trained Model Outside Supervisely</a> •
  <a href="#Screenshot">Screenshot</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmsegmentation/train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/mmsegmentation)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/mmsegmentation/train.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/mmsegmentation/train.png)](https://supervise.ly)

</div>

# Overview

Train MMSegmentation models in Supervisely.

Application key points:
- All Semantic Segmentation models from MM Toolbox
- Use pretrained MMSegmentation models
- Define Train / Validation splits
- Select classes for training
- Define augmentations
- Tune hyperparameters
- Monitor Metrics charts
- Save training artifacts to Team Files
- Support annotations in both bitmap mask and polygon formats

**All MMSegmentation models are supported [(model zoo)](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/model_zoo.md)**

Supported backbones:

- [x] ResNet (CVPR'2016)
- [x] ResNeXt (CVPR'2017)
- [x] [HRNet (CVPR'2019)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/hrnet)
- [x] [ResNeSt (ArXiv'2020)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/resnest)
- [x] [MobileNetV2 (CVPR'2018)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/mobilenet_v2)
- [x] [MobileNetV3 (ICCV'2019)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/mobilenet_v3)
- [x] [Vision Transformer (ICLR'2021)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/vit)
- [x] [Swin Transformer (ICCV'2021)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/swin)
- [x] [Twins (NeurIPS'2021)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/twins)
- [x] [BEiT (ICLR'2022)](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/beit)
- [x] [ConvNeXt (CVPR'2022)](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/convnext)

Supported methods:

- [x] [FCN (CVPR'2015/TPAMI'2017)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/fcn)
- [x] [ERFNet (T-ITS'2017)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/erfnet)
- [x] [UNet (MICCAI'2016/Nat. Methods'2019)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/unet)
- [x] [PSPNet (CVPR'2017)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/pspnet)
- [x] [DeepLabV3 (ArXiv'2017)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/deeplabv3)
- [x] [BiSeNetV1 (ECCV'2018)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/bisenetv1)
- [x] [PSANet (ECCV'2018)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/psanet)
- [x] [DeepLabV3+ (CVPR'2018)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/deeplabv3plus)
- [x] [UPerNet (ECCV'2018)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/upernet)
- [x] [ICNet (ECCV'2018)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/icnet)
- [x] [NonLocal Net (CVPR'2018)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/nonlocal_net)
- [x] [EncNet (CVPR'2018)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/encnet)
- [x] [Semantic FPN (CVPR'2019)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/sem_fpn)
- [x] [DANet (CVPR'2019)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/danet)
- [x] [APCNet (CVPR'2019)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/apcnet)
- [x] [EMANet (ICCV'2019)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/emanet)
- [x] [CCNet (ICCV'2019)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/ccnet)
- [x] [DMNet (ICCV'2019)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/dmnet)
- [x] [ANN (ICCV'2019)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/ann)
- [x] [GCNet (ICCVW'2019/TPAMI'2020)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/gcnet)
- [x] [FastFCN (ArXiv'2019)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/fastfcn)
- [x] [Fast-SCNN (ArXiv'2019)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/fastscnn)
- [x] [ISANet (ArXiv'2019/IJCV'2021)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/isanet)
- [x] [OCRNet (ECCV'2020)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/ocrnet)
- [x] [DNLNet (ECCV'2020)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/dnlnet)
- [x] [PointRend (CVPR'2020)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/point_rend)
- [x] [CGNet (TIP'2020)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/cgnet)
- [x] [BiSeNetV2 (IJCV'2021)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/bisenetv2)
- [x] [STDC (CVPR'2021)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/stdc)
- [x] [SETR (CVPR'2021)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/setr)
- [x] [DPT (ArXiv'2021)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/dpt)
- [x] [Segmenter (ICCV'2021)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/segmenter)
- [x] [SegFormer (NeurIPS'2021)](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/segformer)
- [x] [K-Net (NeurIPS'2021)](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/knet)


# How to Run

### 1. Add [Train MMSegmentation](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmsegmentation/train) app to your team from Ecosystem
<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmsegmentation/train" src="https://i.imgur.com/e2r6ccw.png" width="350px" style='padding-bottom: 10px'/>

### 2. Run app from context menu of the project with annotations (polygon and bitmap only) [(example)](https://ecosystem.supervise.ly/projects/lemons-annotated)
<img src="https://i.imgur.com/XczjaNy.png" width="100%" style='padding-top: 10px'>  

# Demo

<a data-key="sly-embeded-video-link" href="https://youtu.be/R2_8qUw8R_A" data-video-code="R2_8qUw8R_A">
    <img src="https://i.imgur.com/cBMwAlb.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:80%;">
</a>


# How To Use Your Trained Model Outside Supervisely

You can use your trained models outside Supervisely platform without any dependencies on Supervisely SDK. You just need to download config file and model weights (.pth) from Team Files, and then you can build and use the model as a normal model in mmsegmentation. See this [Jupyter Notebook](https://github.com/supervisely-ecosystem/mmsegmentation/blob/main/inference_outside_supervisely.ipynb) for details.


# Screenshot

<img src="https://i.imgur.com/zRHgnfQ.png" width="100%" style='padding-top: 10px'>

# Acknowledgment

This app is based on the great work `MMSegmentation` ([github](https://github.com/open-mmlab/mmsegmentation)). ![GitHub Org's stars](https://img.shields.io/github/stars/open-mmlab/mmsegmentation?style=social)

