
<div align="center" markdown>
<img src="https://i.imgur.com/wQEmCI5.jpg"/>  

# Serve MMSegmentation

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#How-To-Use">How To Use</a> •
  <a href="#Common-apps">Common Apps</a> •
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

Application key points:
- Select from 7 architectures and 40 pretrained models to serve
- Deployed on GPU or CPU


# How to Run

### 1. Add [Serve MMSegmentation](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmsegmentation/serve) to your team
<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmsegmentation/serve" src="https://i.imgur.com/e1SsMJh.png" width="350px" style='padding-bottom: 10px'/>

### 2. Run the application from Plugins & Apps page
<img src="https://i.imgur.com/1uAKeeE.png" width="80%" style='padding-top: 10px'>  

# How to Use

### 1. Select architecture, pretrained model, deploying device and press the **Serve** button
<img src="https://i.imgur.com/yEmmskW.png" width="80%">  

### 2. Wait for the model to deploy
<img src="https://i.imgur.com/FZeg5gT.png" width="80%">  


# Common apps

You can use served model in next Supervisely Applications ⬇️ 
  

- [Train MMSegmentation](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmsegmentation/train) - app allows to play with different inference options, monitor metrics charts in real time, and save training artifacts to Team Files.
   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmsegmentation/train" src="https://i.imgur.com/e2r6ccw.png" width="350px"/>
    
- [Apply NN to images project ](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fproject-dataset) - app allows to play with different inference options and visualize predictions in real time.  Once you choose inference settings you can apply model to all images in your project to visually analyse predictions and perform automatic data pre-labeling.   
   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/project-dataset" src="https://i.imgur.com/M2Tp8lE.png" width="350px"/> 

- [NN Image Labeling](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) - integrate any deployd NN to Supervisely Image Labeling UI. Configure inference settings and model output classes. Press `Apply` button (or use hotkey) and detections with their confidences will immediately appear on the image. 
   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/annotation-tool" src="https://i.imgur.com/hYEucNt.png" width="350px"/>

# Acknowledgment

This app is based on the great work `MMSegmentation` ([github](https://github.com/open-mmlab/mmsegmentation)). ![GitHub Org's stars](https://img.shields.io/github/stars/open-mmlab/mmsegmentation?style=social)

