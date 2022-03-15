
<div align="center" markdown>
<img src="https://i.imgur.com/UbiaKsm.jpg"/>  

# Serve MMSegmentation

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#How-To-Use">How To Use</a> •
  <a href="#Related-apps">Related Apps</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmsegmentation/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/mmsegmentation)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/mmsegmentation/serve&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/mmsegmentation/serve&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/mmsegmentation/serve&counter=runs&label=runs&123)](https://supervise.ly)

</div>

# Overview

Serve MMSegmentation model as Supervisely Application. MMSegmentation is an open source semantic segmentation toolbox based on PyTorch. Learn more about MMSegmentation and available models [here](https://github.com/open-mmlab/mmsegmentation).

Model serving allows to apply model to image (URL, local file, Supervisely image id) with 2 modes (full image, image ROI). Also app sources can be used as example how to use downloaded model weights outside Supervisely.

Application key points:
- Serve custom and MMSegmentation models
- Deployed on GPU

## Available models

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


# How to Run

### 1. Add [Serve MMSegmentation](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmsegmentation/serve) app to your team from Ecosystem
<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmsegmentation/serve" src="https://i.imgur.com/e1SsMJh.png" width="350px" style='padding-bottom: 10px'/>

### 2. Run the application from Plugins & Apps page
<img src="https://i.imgur.com/1uAKeeE.png" width="80%" style='padding-top: 10px'>  

# How to Use

## Pretrained models

### 1. Select architecture, pretrained model and press the **Serve** button

<img src="https://i.imgur.com/yEmmskW.png" style="width: 100%"/>


### 2. Wait for the model to deploy
<img src="https://i.imgur.com/FZeg5gT.png" width="100%">  


## Custom models

### Model and directory structure must be acquired via [Train MMSegmentation](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmsegmentation/train) app or manually created with the same directory structure

<img src="https://github.com/supervisely-ecosystem/mmsegmentation/releases/download/v0.0.1/custom_weights_guide-min.gif" style="width: 100%"/>

# Related apps

You can use served model in next Supervisely Applications ⬇️ 

- [Train MMSegmentation](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmsegmentation/train) - app allows to play with different inference options, monitor metrics charts in real time, and save training artifacts to Team Files.
   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmsegmentation/train" src="https://i.imgur.com/e2r6ccw.png" width="350px"/>
    
- [Apply NN to images project ](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fproject-dataset) - app allows to play with different inference options and visualize predictions in real time.  Once you choose inference settings you can apply model to all images in your project to visually analyse predictions and perform automatic data pre-labeling.   
   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/project-dataset" src="https://i.imgur.com/M2Tp8lE.png" width="350px"/> 

- [NN Image Labeling](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) - integrate any deployd NN to Supervisely Image Labeling UI. Configure inference settings and model output classes. Press `Apply` button (or use hotkey) and detections with their confidences will immediately appear on the image. 
   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/annotation-tool" src="https://i.imgur.com/hYEucNt.png" width="350px"/>

# Acknowledgment

This app is based on the great work `MMSegmentation` ([github](https://github.com/open-mmlab/mmsegmentation)). ![GitHub Org's stars](https://img.shields.io/github/stars/open-mmlab/mmsegmentation?style=social)

