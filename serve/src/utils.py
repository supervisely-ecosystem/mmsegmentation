import supervisely as sly
import yaml
import pkg_resources
import os
import errno
import sly_globals as g
import cv2
import sys
from mmcv import Config
from mmseg.datasets import *
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.models import build_segmentor
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import load_checkpoint

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def inference_image_path(image_path, context, state, app_logger):
    app_logger.debug("Input path", extra={"path": image_path})

    img = cv2.imread(image_path)
    raw_result = inference_segmentor(g.model, img)[0]

    labels = []
    classes = [obj["title"] for obj in g.meta.obj_classes.to_json()]
    for idx, class_name in enumerate(classes):
        class_mask = raw_result == idx
        obj_class = g.meta.get_obj_class(class_name)
        label = sly.Label(sly.Bitmap(class_mask), obj_class)
        labels.append(label)

    ann = sly.Annotation(img_size=raw_result.shape, labels=labels, )
    ann_json = ann.to_json()

    return ann_json

def get_pretrained_models(return_metrics=False):
    model_yamls = sly.json.load_json_file(os.path.join(g.root_source_path, "models", "model_meta.json"))
    model_config = {}
    all_metrics = []
    for model_meta in model_yamls:
        with open(os.path.join(g.configs_dir, model_meta["yml_file"]), "r") as stream:
            model_info = yaml.safe_load(stream)
            model_config[model_meta["model_name"]] = {}
            # model_config[model_meta["model_name"]]["code_url"] = model_info["Collections"][0]["Code"]["URL"]
            # model_config[model_meta["model_name"]]["paper_title"] = model_info["Collections"][0]["Paper"]["Title"]
            # model_config[model_meta["model_name"]]["paper_url"] = model_info["Collections"][0]["Paper"]["URL"]
            model_config[model_meta["model_name"]]["checkpoints"] = []
            model_config[model_meta["model_name"]]["paper_from"] = model_meta["paper_from"]
            model_config[model_meta["model_name"]]["year"] = model_meta["year"]
            mmseg_ver = pkg_resources.get_distribution("mmsegmentation").version
            # TODO: change link to current version of package
            model_config[model_meta["model_name"]]["config_url"] = f"https://github.com/open-mmlab/mmsegmentation/tree/v{mmseg_ver}/configs/" + model_meta["yml_file"].split("/")[0]
            checkpoint_keys = []
            for model in model_info["Models"]:
                checkpoint_info = {}
                checkpoint_info["name"] = model["Name"]
                checkpoint_info["backbone"] = model["Metadata"]["backbone"]
                try:
                    checkpoint_info["inference_time"] = model["Metadata"]["inference time (ms/im)"][0]["value"]
                except KeyError:
                    checkpoint_info["inference_time"] = "-"
                checkpoint_info["crop_size"] = model["Metadata"]["crop size"]
                checkpoint_info["lr_schd"] = model["Metadata"]["lr schd"]
                try:
                    checkpoint_info["training_memory"] = model["Metadata"]["Training Memory (GB)"]
                except KeyError:
                    checkpoint_info["training_memory"] = "-"
                checkpoint_info["config_file"] = model["Config"]
                checkpoint_info["dataset"] = model["Results"][0]["Dataset"]
                for metric_name, metric_val in model["Results"][0]["Metrics"].items():
                    if metric_name not in all_metrics:
                        all_metrics.append(metric_name)
                    checkpoint_info[metric_name] = metric_val
                checkpoint_info["weights"] = model["Weights"]
                for key in checkpoint_info.keys():
                    checkpoint_keys.append(key)
                model_config[model_meta["model_name"]]["checkpoints"].append(checkpoint_info)
            model_config[model_meta["model_name"]]["all_keys"] = checkpoint_keys
    if return_metrics:
        return model_config, all_metrics
    return model_config


def get_table_columns(metrics):
    columns = [
        {"key": "name", "title": " ", "subtitle": None},
        {"key": "backbone", "title": "Backbone", "subtitle": None},
        {"key": "dataset", "title": "Dataset", "subtitle": None},
        {"key": "inference_time", "title": "Inference time", "subtitle": "(ms/im)"},
        {"key": "crop_size", "title": "Input size", "subtitle": "(H, W)"},
        {"key": "lr_schd", "title": "LR scheduler", "subtitle": "steps"},
        {"key": "training_memory", "title": "Training memory", "subtitle": "GB"},
    ]
    for metric in metrics:
        columns.append({"key": metric, "title": f"{metric} score", "subtitle": None})
    return columns


def download_sly_file(remote_path, local_path):
    file_info = g.api.file.get_info_by_path(g.TEAM_ID, remote_path)
    if file_info is None:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), remote_path)
    g.api.file.download(g.TEAM_ID, remote_path, local_path, g.my_app.cache)

    sly.logger.info(f"{remote_path} has been successfully downloaded",
                    extra={"weights": local_path})


def download_custom_config(state):
    weights_remote_dir = os.path.dirname(state["weightsPath"])
    g.model_config_local_path = os.path.join(g.models_configs_dir, 'config.py')

    config_remote_dir = os.path.join(weights_remote_dir, f'config.py')
    if g.api.file.exists(g.TEAM_ID, config_remote_dir):
        download_sly_file(config_remote_dir, g.model_config_local_path)


def download_weights(state):
    if state["weightsInitialization"] == "custom":
        weights_path_remote = state["weightsPath"]
        if not weights_path_remote.endswith(".pth"):
            raise ValueError(f"Weights file has unsupported extension {sly.fs.get_file_ext(weights_path_remote)}. "
                             f"Supported: '.pth'")

        g.local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_path_remote))
        if sly.fs.file_exists(g.local_weights_path):
            os.remove(g.local_weights_path)

        download_sly_file(weights_path_remote, g.local_weights_path)
        download_custom_config(state)

    else:
        checkpoints_by_model = get_pretrained_models()[state['pretrainedModel']]["checkpoints"]
        selected_model = next(item for item in checkpoints_by_model
                              if item["name"] == state["selectedModel"][state["pretrainedModel"]])

        weights_url = selected_model.get('weights')
        config_file = selected_model.get('config_file')
        g.dataset = selected_model.get("dataset")
        if weights_url is not None:
            g.local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_url))
            g.model_config_local_path = os.path.join(g.root_source_path, config_file)
            # TODO: check that pretrained weights are exist on remote server
            if sly.fs.file_exists(g.local_weights_path) is False:
                os.makedirs(os.path.dirname(g.local_weights_path), exist_ok=True)
                sly.fs.download(weights_url, g.local_weights_path, g.my_app.cache)
            sly.logger.info("Pretrained weights has been successfully downloaded",
                            extra={"weights": g.local_weights_path})

def init_model_and_cfg(state):
    g.cfg = Config.fromfile(g.model_config_local_path)
    g.cfg.model.pretrained = None
    g.cfg.model.train_cfg = None
    model = build_segmentor(g.cfg.model, test_cfg=g.cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, g.local_weights_path, map_location='cpu')
    if state["weightsInitialization"] == "custom":
        classes = g.cfg.checkpoint_config.meta.CLASSES
        palette = g.cfg.checkpoint_config.meta.PALETTE
    else:
        dataset_class_name = g.cfg.dataset_type
        classes = str_to_class(dataset_class_name).CLASSES
        palette = str_to_class(dataset_class_name).PALETTE

    model.CLASSES = classes
    model.PALETTE = palette
    model.cfg = g.cfg  # save the config in the model for convenience
    model.to(g.device)
    model.eval()
    model = revert_sync_batchnorm(model)
    g.model = model

    obj_classes = [sly.ObjClass(name, sly.Bitmap, color) for name, color in zip(classes, palette)]
    g.meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))