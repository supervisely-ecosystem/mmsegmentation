import os
import shutil
import sys

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pkg_resources
import torch
import yaml
from dotenv import load_dotenv
from mmcv import Config
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import load_checkpoint
from mmseg.apis.inference import inference_segmentor
from mmseg.datasets import *
from mmseg.models import build_segmentor

import supervisely as sly
from serve.src import utils
from supervisely.nn.artifacts.mmsegmentation import MMSegmentation
from supervisely.app.widgets import (
    CustomModelsSelector,
    PretrainedModelsSelector,
    RadioTabs,
    Widget,
)
from supervisely.io.fs import silent_remove
import workflow as w

root_source_path = str(Path(__file__).parents[2])
app_source_path = str(Path(__file__).parents[1])
load_dotenv(os.path.join(app_source_path, "local.env"))
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()
team_id = sly.env.team_id()

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
    def initialize_custom_gui(self) -> Widget:
        """Create custom GUI layout for model selection. This method is called once when the application is started."""
        models = self.get_models()
        filtered_models = utils.filter_models_structure(models)
        self.pretrained_models_table = PretrainedModelsSelector(filtered_models)
        sly_mmseg = MMSegmentation(team_id)
        custom_models = sly_mmseg.get_list()
        self.custom_models_table = CustomModelsSelector(
            team_id,
            custom_models,
            show_custom_checkpoint_path=True,
            custom_checkpoint_task_types=["semantic segmentation"],
        )

        self.model_source_tabs = RadioTabs(
            titles=["Pretrained models", "Custom models"],
            descriptions=["Publicly available models", "Models trained by you in Supervisely"],
            contents=[self.pretrained_models_table, self.custom_models_table],
        )
        return self.model_source_tabs

    def get_params_from_gui(self) -> dict:
        model_source = self.model_source_tabs.get_active_tab()
        self.device = self.gui.get_device()
        if model_source == "Pretrained models":
            model_params = self.pretrained_models_table.get_selected_model_params()
        elif model_source == "Custom models":
            model_params = self.custom_models_table.get_selected_model_params()
            if self.custom_models_table.use_custom_checkpoint_path():
                checkpoint_path = self.custom_models_table.get_custom_checkpoint_path()
                model_params["config_url"] = (
                    f"{os.path.dirname(checkpoint_path).rstrip('/')}/config.py"
                )
                file_info = api.file.exists(team_id, model_params["config_url"])
                if file_info is None:
                    raise FileNotFoundError(
                        f"Config file not found: {model_params['config_url']}. "
                        "Config should be placed in the same directory as the checkpoint file."
                    )

        self.selected_model_name = model_params.get("arch_type")
        self.checkpoint_name = model_params.get("checkpoint_name")
        self.task_type = model_params.get("task_type")

        deploy_params = {
            "device": self.device,
            **model_params,
        }
        return deploy_params

    def load_model_meta(
        self, model_source: str, cfg: Config, checkpoint_name: str = None, arch_type: str = None
    ):
        def set_common_meta(classes, palette):
            obj_classes = [
                sly.ObjClass(name, sly.Bitmap, color) for name, color in zip(classes, palette)
            ]
            self.checkpoint_name = checkpoint_name
            self.dataset_name = cfg.dataset_type
            self.class_names = classes
            self._model_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))
            self._get_confidence_tag_meta()

        if model_source == "Custom models":
            self.selected_model_name = cfg.pretrained_model
            classes = cfg.checkpoint_config.meta.CLASSES
            palette = cfg.checkpoint_config.meta.PALETTE
            set_common_meta(classes, palette)

        elif model_source == "Pretrained models":
            self.selected_model_name = arch_type
            dataset_class_name = cfg.dataset_type
            classes = str_to_class(dataset_class_name).CLASSES
            palette = str_to_class(dataset_class_name).PALETTE
            set_common_meta(classes, palette)

        self.model.CLASSES = classes
        self.model.PALETTE = palette

    def load_model(
        self,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        model_source: Literal["Pretrained models", "Custom models"],
        task_type: Literal["semantic segmentation"],
        checkpoint_name: str,
        checkpoint_url: str,
        config_url: str,
        arch_type: str = None,
    ):
        """
        Load model method is used to deploy model.

        :param model_source: Specifies whether the model is pretrained or custom.
        :type model_source: Literal["Pretrained models", "Custom models"]
        :param device: The device on which the model will be deployed.
        :type device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        :param task_type: The type of task the model is designed for.
        :type task_type: Literal["semantic segmentation"]
        :param checkpoint_name: The name of the checkpoint from which the model is loaded.
        :type checkpoint_name: str
        :param checkpoint_url: The URL where the model checkpoint can be downloaded.
        :type checkpoint_url: str
        :param config_url: The URL where the model config can be downloaded.
        :type config_url: str
        :param arch_type: The architecture type of the model.
        :type arch_type: str
        """
        self.device = device
        self.task_type = task_type

        local_weights_path = os.path.join(self.model_dir, checkpoint_name)
        if model_source == "Pretrained models":
            if not sly.fs.file_exists(local_weights_path):
                self.download(
                    src_path=checkpoint_url,
                    dst_path=local_weights_path,
                )
            local_config_path = os.path.join(root_source_path, config_url)
        else:
            self.download(
                src_path=checkpoint_url,
                dst_path=local_weights_path,
            )
            local_config_path = os.path.join(configs_dir, "custom", "config.py")
            if sly.fs.file_exists(local_config_path):
                silent_remove(local_config_path)
            self.download(
                src_path=config_url,
                dst_path=local_config_path,
            )
            if not sly.fs.file_exists(local_config_path):
                raise FileNotFoundError(
                    f"Config file not found: {config_url}. "
                    "Config should be placed in the same directory as the checkpoint file."
                )
        try:
            cfg = Config.fromfile(local_config_path)
            cfg.model.pretrained = None
            cfg.model.train_cfg = None

            self.model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
            checkpoint = load_checkpoint(self.model, local_weights_path, map_location="cpu")

            self.load_model_meta(model_source, cfg, checkpoint_name, arch_type)

            self.model.cfg = cfg  # save the config in the model for convenience
            self.model.to(device)
            # -------------------------------------- Add Workflow Input -------------------------------------- #
            sly.logger.debug("Workflow: Start processing Input")
            if model_source == "Custom models":
                sly.logger.debug("Workflow: Custom model detected")
                w.workflow_input(api, checkpoint_url)
            else:
                sly.logger.debug("Workflow: Pretrained model detected. No need to set Input")
            sly.logger.debug("Workflow: Finish processing Input")
            # ----------------------------------------------- - ---------------------------------------------- #
            self.model.eval()
            self.model = revert_sync_batchnorm(self.model)

            # Set checkpoint info
            if model_source == "Pretrained models":
                custom_checkpoint_path = None
            else:
                custom_checkpoint_path = checkpoint_url
                file_id = self.api.file.get_info_by_path(self.team_id, checkpoint_url).id
                checkpoint_url = self.api.file.get_url(file_id)
            if arch_type is None:
                arch_type = self.parse_arch_type(cfg)

            self.checkpoint_info = sly.nn.inference.CheckpointInfo(
                checkpoint_name=checkpoint_name,
                model_name=self.selected_model_name,
                architecture=arch_type,
                checkpoint_url=checkpoint_url,
                custom_checkpoint_path=custom_checkpoint_path,
                model_source=model_source,
            )

        except KeyError as e:
            raise KeyError(f"Error loading config file: {local_config_path}. Error: {e}")

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
                weights_path, config_path = self.download_pretrained_files(
                    selected_model, model_dir
                )
            elif model_source == "Custom models":
                custom_weights_link = self.gui.get_custom_link()
                weights_path, config_path = self.download_custom_files(
                    custom_weights_link, model_dir
                )
            sly.logger.debug(f"Model source if GUI is not None: {model_source}")
        else:
            # for local debug only
            model_source = "Pretrained models"
            weights_path, config_path = self.download_pretrained_files(
                selected_checkpoint, model_dir
            )
            sly.logger.debug(f"Model source if GUI is None: {model_source}")

        cfg = Config.fromfile(config_path)
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
        checkpoint = load_checkpoint(model, weights_path, map_location="cpu")
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

        obj_classes = [
            sly.ObjClass(name, sly.Bitmap, color) for name, color in zip(classes, palette)
        ]
        self._model_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def get_info(self) -> dict:
        info = super().get_info()
        info["model_name"] = self.selected_model_name
        info["checkpoint_name"] = self.checkpoint_name
        info["pretrained_on_dataset"] = self.dataset_name
        info["device"] = self.device
        return info

    def get_models(self):
        model_yamls = sly.json.load_json_file(models_meta_path)
        model_config = {}
        for model_meta in model_yamls:
            mmseg_ver = pkg_resources.get_distribution("mmsegmentation").version
            model_yml_url = f"https://github.com/open-mmlab/mmsegmentation/tree/v{mmseg_ver}/configs/{model_meta['yml_file']}"
            model_yml_local = os.path.join(configs_dir, model_meta["yml_file"])
            with open(model_yml_local, "r") as stream:
                model_info = yaml.safe_load(stream)
                model_config[model_meta["model_name"]] = {}
                model_config[model_meta["model_name"]]["checkpoints"] = []
                model_config[model_meta["model_name"]]["paper_from"] = model_meta["paper_from"]
                model_config[model_meta["model_name"]]["year"] = model_meta["year"]
                model_config[model_meta["model_name"]]["config_url"] = os.path.dirname(
                    model_yml_url
                )
                for model in model_info["Models"]:
                    checkpoint_info = OrderedDict()
                    checkpoint_info["Model"] = model["Name"]
                    checkpoint_info["Backbone"] = model["Metadata"]["backbone"]
                    checkpoint_info["Method"] = model["In Collection"]
                    checkpoint_info["Dataset"] = model["Results"][0]["Dataset"]
                    try:
                        checkpoint_info["Inference Time (ms/im)"] = model["Metadata"][
                            "inference time (ms/im)"
                        ][0]["value"]
                    except KeyError:
                        checkpoint_info["Inference Time (ms/im)"] = "-"
                    checkpoint_info["Input Size (H, W)"] = model["Metadata"]["crop size"]
                    checkpoint_info["LR scheduler (steps)"] = model["Metadata"]["lr schd"]
                    try:
                        checkpoint_info["Memory (Training, GB)"] = model["Metadata"][
                            "Training Memory (GB)"
                        ]
                    except KeyError:
                        checkpoint_info["Memory (Training, GB)"] = "-"
                    for metric_name, metric_val in model["Results"][0]["Metrics"].items():
                        checkpoint_info[metric_name] = metric_val
                    # checkpoint_info["config_file"] = os.path.join(f"https://github.com/open-mmlab/mmsegmentation/tree/v{mmseg_ver}", model["Config"])
                    checkpoint_info["meta"] = {
                        "task_type": None,
                        "arch_type": None,
                        "arch_link": None,
                        "weights_url": model["Weights"],
                        "config_url": os.path.join(root_source_path, model["Config"]),
                    }
                    model_config[model_meta["model_name"]]["checkpoints"].append(checkpoint_info)
        return model_config

    def parse_arch_type(self, cfg: Config) -> str:
        try:
            arch_type = cfg.model.backbone.type
            try:
                arch_type += f"_{cfg.model.backbone.arch}"
            except:
                pass
            return arch_type
        except Exception as e:
            sly.logger.warning(f"Error parsing model name: {e}")

    def get_classes(self) -> List[str]:
        return self.class_names  # e.g. ["cat", "dog", ...]

    def predict(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[sly.nn.PredictionSegmentation]:

        segmented_image = inference_segmentor(self.model, image_path)[0]

        return [sly.nn.PredictionSegmentation(segmented_image)]


if sly.is_production():
    sly.logger.info(
        "Script arguments",
        extra={
            "context.teamId": sly.env.team_id(),
            "context.workspaceId": sly.env.workspace_id(),
        },
    )

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
