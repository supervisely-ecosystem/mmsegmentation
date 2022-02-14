
<div align="center" markdown>

<img src="https://i.imgur.com/5QJrX7k.png"/>  

# Serve MMSegmentation

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Demo">Demo</a> •
  <a href="#Screenshot">Screenshot</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmsegmentation/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/mmsegmentation/serve)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/mmsegmentation/serve&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/mmsegmentation/serve&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/mmsegmentation/serve&counter=runs&label=runs&123)](https://supervise.ly)

</div>

# Overview

Serve MMSegmentation model as Supervisely Application. MMSegmentation is an open source semantic segmentation toolbox based on PyTorch. Learn more about MMSegmentation and available models [here](https://github.com/open-mmlab/mmsegmentation).

Application key points:
- Select from 7 architectures and many pretrained models to serve
- Deployed on GPU or CPU


# How to Run

### 1. Add [Serve Detectron2](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/detectron2/supervisely/instance_segmentation/serve) to your team
<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/detectron2/supervisely/instance_segmentation/serve" src="https://imgur.com/jKrRF7p.png" width="350px" style='padding-bottom: 10px'/>

### 2. Choose architecture, pretrained model, deploying device and press the **Run** button
<img src="https://imgur.com/DLDYMbk.png" width="80%" style='padding-top: 10px'>  

### 3. Wait for the model to deploy
<img src="https://imgur.com/KFdwTER.png" width="80%">  


# Common apps

You can use served model in next Supervisely Applications ⬇️ 
  

- [Train MMSegmentation](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmsegmentation/train) - app allows to play with different inference options, monitor metrics charts in real time, and save training artifacts to Team Files.
   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmsegmentation/train" src="https://i.imgur.com/FvszlJJ.png" width="350px"/>

# Acknowledgment

This app is based on the great work `MMSegmentation` ([github](https://github.com/open-mmlab/mmsegmentation)). ![GitHub Org's stars](https://img.shields.io/github/stars/open-mmlab/mmsegmentation?style=social)

