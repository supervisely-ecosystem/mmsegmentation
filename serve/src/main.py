import os
import shutil
import sys
try:
    from typing import Literal
except:
    from typing_extensions import Literal
from typing import List, Any, Dict, Union
from pathlib import Path
import numpy as np
import yaml
from dotenv import load_dotenv
import torch
import supervisely as sly
import supervisely.app.widgets as Widgets
from supervisely.nn.inference import FilesContext
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

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

# TODO:
mmseg_ver = pkg_resources.get_distribution("mmsegmentation").version
mmseg_repo_path = os.path.join(root_source_path, f"mmsegmentation-{mmseg_ver}")
configs_dir = os.path.join(mmseg_repo_path, "configs")
# if os.path.isdir(mmseg_repo_path):
#     if os.path.isdir(configs_dir):
#         shutil.rmtree(configs_dir)
#     sly.logger.info(f"Getting model configs of current mmsegmentation version {mmseg_ver}...")
#     shutil.copytree(f"/tmp/mmseg/mmsegmentation-{mmseg_ver}/configs", configs_dir)
models_cnt = len(os.listdir(configs_dir)) - 1
sly.logger.info(f"Found {models_cnt} models in {configs_dir} directory.")


class MMSegmentationModel(sly.nn.inference.SemanticSegmentation):
    def load_on_device(
        self,
        files_context: FilesContext, # from GUI
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ) -> None:
        config_file = files_context.get("config_file")
        cfg = Config.fromfile(config_file.location)
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        checkpoint = load_checkpoint(model, files_context.get("weights_file").location, map_location='cpu')
        model_source = self.gui.get_model_source()
        if model_source == "Custom weights":
            classes = cfg.checkpoint_config.meta.CLASSES
            palette = cfg.checkpoint_config.meta.PALETTE
        elif model_source == "Pretrained models":
            dataset_class_name = cfg.dataset_type
            classes = str_to_class(dataset_class_name).CLASSES
            palette = str_to_class(dataset_class_name).PALETTE

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

    def get_models(self) -> Union[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]]:
        models_meta_path = os.path.join(root_source_path, "models", "model_meta.json")
        return self.get_pretrained_models(models_meta_path)

    def get_pretrained_models(self, models_meta_path):
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
                    checkpoint_info["config_file"] = os.path.join(mmseg_repo_path, model["Config"])
                    checkpoint_info["weights_file"] = model["Weights"]
                    model_config[model_meta["model_name"]]["checkpoints"].append(checkpoint_info)
        return model_config

    def get_custom_files(custom_link):
        weights_remote_dir = os.path.dirname(custom_link)
        return {
            "weights_file": custom_link,
            "config_file": os.path.join(weights_remote_dir, 'config.py')
        }

    def predict(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[sly.nn.PredictionSegmentation]:

        segmented_image = inference_segmentor(self.model, image_path)[0]

        return [sly.nn.PredictionSegmentation(segmented_image)]

sly.logger.info("Script arguments", extra={
    "context.teamId": sly.env.team_id(),
    "context.workspaceId": sly.env.workspace_id(),
})

# TODO: get from GUI
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

m = MMSegmentationModel(use_gui=True)

if sly.is_production():
    # this code block is running on Supervisely platform in production
    # just ignore it during development
    m.serve()
else:
    # for local development and debugging
    # TODO: how to be with GUI?

    # location = 
    # m.load_on_device(m.get_context(location), device)    
    image_path = "./demo_data/image_01.jpg"
    results = m.predict(image_path, {})
    vis_path = "./demo_data/image_01_prediction.jpg"
    m.visualize(results, image_path, vis_path, thickness=0)
    print(f"predictions and visualization have been saved: {vis_path}")
