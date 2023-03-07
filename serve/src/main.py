import os
import shutil
import sys
try:
    from typing import Literal
except:
    from typing_extensions import Literal
from typing import List, Any, Dict
from pathlib import Path
import numpy as np
import yaml
from dotenv import load_dotenv
import torch
import supervisely as sly
import supervisely.app.widgets as Widgets
import supervisely.nn.inference.gui as GUI
import pkg_resources
from collections import OrderedDict
from mmcv import Config
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
from mmseg.apis.inference import inference_segmentor
from mmseg.datasets import *

root_source_path = str(Path(__file__).parents[2])
app_source_path = str(Path(__file__).parents[1])
load_dotenv(os.path.join(app_source_path, "local.env"))
load_dotenv(os.path.expanduser("~/supervisely.env"))

use_gui_for_local_debug = bool(int(os.environ.get("USE_GUI", "1")))

models_meta_path = os.path.join(root_source_path, "models", "model_meta.json")

# for local debug
selected_checkpoint = None 
selected_model_name = None

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

configs_dir = os.path.join(root_source_path, "configs")
mmseg_ver = pkg_resources.get_distribution("mmsegmentation").version
if os.path.isdir(f"/tmp/mmseg/mmsegmentation-{mmseg_ver}"):
    if os.path.isdir(configs_dir):
        shutil.rmtree(configs_dir)
    sly.logger.info(f"Getting model configs of current mmsegmentation version {mmseg_ver}...")
    shutil.copytree(f"/tmp/mmseg/mmsegmentation-{mmseg_ver}/configs", configs_dir)
    models_cnt = len(os.listdir(configs_dir)) - 1
    sly.logger.info(f"Found {models_cnt} models in {configs_dir} directory.")


class MMSegmentationModel(sly.nn.inference.SemanticSegmentation):
    def load_on_device(
        self,
        model_dir: str, 
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ) -> None:
        self.device = device
        if self.gui is not None:
            model_source = self.gui.get_model_source()
            if model_source == "Pretrained models":
                selected_model = self.gui.get_checkpoint_info()
                weights_path, config_path = self.download_pretrained_files(selected_model, model_dir)
            elif model_source == "Custom models":
                custom_weights_link = self.gui.get_custom_link()
                weights_path, config_path = self.download_custom_files(custom_weights_link, model_dir)
        else:
            # for local debug only
            model_source = "Pretrained models"
            weights_path, config_path = self.download_pretrained_files(selected_checkpoint, model_dir)
        cfg = Config.fromfile(config_path)
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        checkpoint = load_checkpoint(model, weights_path, map_location='cpu')
        if model_source == "Custom models":
            classes = cfg.checkpoint_config.meta.CLASSES
            palette = cfg.checkpoint_config.meta.PALETTE
            self.selected_model_name = cfg.pretrained_model
            self.checkpoint_name = "custom"
            self.dataset_name = "custom"
        elif model_source == "Pretrained models":
            dataset_class_name = cfg.dataset_type
            classes = str_to_class(dataset_class_name).CLASSES
            palette = str_to_class(dataset_class_name).PALETTE
            if self.gui is not None:
                self.selected_model_name = list(self.gui.get_model_info().keys())[0]
                checkpoint_info = self.gui.get_checkpoint_info()
                self.checkpoint_name = checkpoint_info["Name"]
                self.dataset_name = checkpoint_info["Dataset"]
            else:
                self.selected_model_name = selected_model_name
                self.checkpoint_name = selected_checkpoint["Name"]
                self.dataset_name = dataset_name

        model.CLASSES = classes
        model.PALETTE = palette
        model.cfg = cfg  # save the config in the model for convenience
        model.to(device)
        model.eval()
        model = revert_sync_batchnorm(model)
        self.model = model
        self.class_names = classes

        obj_classes = [sly.ObjClass(name, sly.Bitmap, color) for name, color in zip(classes, palette)]
        self._model_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def get_classes(self) -> List[str]:
        return self.class_names  # e.g. ["cat", "dog", ...]

    def get_info(self) -> dict:
        info = super().get_info()
        info["model_name"] = self.selected_model_name
        info["checkpoint_name"] = self.checkpoint_name
        info["pretrained_on_dataset"] = self.dataset_name
        info["device"] = self.device
        return info

    def get_models(self, add_links=False):
        model_yamls = sly.json.load_json_file(models_meta_path)
        model_config = {}
        for model_meta in model_yamls:
            mmseg_ver = pkg_resources.get_distribution("mmsegmentation").version
            model_yml_url = f"https://github.com/open-mmlab/mmsegmentation/tree/v{mmseg_ver}/configs/{model_meta['yml_file']}"
            model_yml_local = os.path.join(configs_dir, model_meta['yml_file'])
            with open(model_yml_local, "r") as stream:
                model_info = yaml.safe_load(stream)
                model_config[model_meta["model_name"]] = {}
                model_config[model_meta["model_name"]]["checkpoints"] = []
                model_config[model_meta["model_name"]]["paper_from"] = model_meta["paper_from"]
                model_config[model_meta["model_name"]]["year"] = model_meta["year"]
                model_config[model_meta["model_name"]]["config_url"] = os.path.dirname(model_yml_url)
                for model in model_info["Models"]:
                    checkpoint_info = OrderedDict()
                    checkpoint_info["Name"] = model["Name"]
                    checkpoint_info["Backbone"] = model["Metadata"]["backbone"]
                    checkpoint_info["Method"] = model["In Collection"]
                    checkpoint_info["Dataset"] = model["Results"][0]["Dataset"]
                    try:
                        checkpoint_info["Inference Time (ms/im)"] = model["Metadata"]["inference time (ms/im)"][0]["value"]
                    except KeyError:
                        checkpoint_info["Inference Time (ms/im)"] = "-"
                    checkpoint_info["Input Size (H, W)"] = model["Metadata"]["crop size"]
                    checkpoint_info["LR scheduler (steps)"] = model["Metadata"]["lr schd"]
                    try:
                        checkpoint_info["Memory (Training, GB)"] = model["Metadata"]["Training Memory (GB)"]
                    except KeyError:
                        checkpoint_info["Memory (Training, GB)"] = "-"
                    for metric_name, metric_val in model["Results"][0]["Metrics"].items():
                        checkpoint_info[metric_name] = metric_val
                    #checkpoint_info["config_file"] = os.path.join(f"https://github.com/open-mmlab/mmsegmentation/tree/v{mmseg_ver}", model["Config"])
                    if add_links:
                        checkpoint_info["config_file"] = os.path.join(root_source_path, model["Config"])
                        checkpoint_info["weights_file"] = model["Weights"]
                    model_config[model_meta["model_name"]]["checkpoints"].append(checkpoint_info)
        return model_config

    def download_pretrained_files(self, selected_model: Dict[str, str], model_dir: str):
        models = self.get_models(add_links=True)
        if self.gui is not None:
            model_name = list(self.gui.get_model_info().keys())[0]
        else:
            # for local debug only
            model_name = selected_model_name
        full_model_info = selected_model
        for model_info in models[model_name]["checkpoints"]:
            if model_info["Name"] == selected_model["Name"]:
                full_model_info = model_info
        weights_ext = sly.fs.get_file_ext(full_model_info["weights_file"])
        config_ext = sly.fs.get_file_ext(full_model_info["config_file"])
        weights_dst_path = os.path.join(model_dir, f"{selected_model['Name']}{weights_ext}")
        if not sly.fs.file_exists(weights_dst_path):
            self.download(
                src_path=full_model_info["weights_file"], 
                dst_path=weights_dst_path
            )
        config_path = self.download(
            src_path=full_model_info["config_file"], 
            dst_path=os.path.join(model_dir, f"config{config_ext}")
        )
        
        return weights_dst_path, config_path

    def download_custom_files(self, custom_link: str, model_dir: str):
        weight_filename = os.path.basename(custom_link)
        weights_dst_path = os.path.join(model_dir, weight_filename)
        if not sly.fs.file_exists(weights_dst_path):
            self.download(
                src_path=custom_link,
                dst_path=weights_dst_path,
            )
        config_path = self.download(
            src_path=os.path.join(os.path.dirname(custom_link), 'config.py'),
            dst_path=os.path.join(model_dir, 'config.py'),
        )
        
        return weights_dst_path, config_path

    def predict(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[sly.nn.PredictionSegmentation]:

        segmented_image = inference_segmentor(self.model, image_path)[0]

        return [sly.nn.PredictionSegmentation(segmented_image)]


if sly.is_production():
    sly.logger.info("Script arguments", extra={
        "context.teamId": sly.env.team_id(),
        "context.workspaceId": sly.env.workspace_id(),
    })

m = MMSegmentationModel(use_gui=True)

if sly.is_production() or use_gui_for_local_debug is True:
    # this code block is running on Supervisely platform in production
    # just ignore it during development
    m.serve()
else:
    # for local development and debugging without GUI
    models = m.get_models(add_links=True)
    selected_model_name = "Segmenter"
    dataset_name = "ADE20K"
    selected_checkpoint = models[selected_model_name]["checkpoints"][0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    m.load_on_device(m.model_dir, device)
    image_path = "./demo_data/image_01.jpg"
    results = m.predict(image_path, {})
    vis_path = "./demo_data/image_01_prediction.jpg"
    m.visualize(results, image_path, vis_path, thickness=0)
    print(f"predictions and visualization have been saved: {vis_path}")

