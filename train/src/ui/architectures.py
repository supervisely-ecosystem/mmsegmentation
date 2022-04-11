import errno
import os
import requests
import yaml
import pkg_resources
import sly_globals as g
import supervisely as sly
from supervisely.app.v1.widgets.progress_bar import ProgressBar
import init_default_cfg as init_dc
from mmcv import Config

cfg = None

def init(data, state):
    state['pretrainedModel'] = 'ConvNeXt'
    data["pretrainedModels"], metrics = get_pretrained_models(return_metrics=True)
    model_select_info = []
    for model_name, params in data["pretrainedModels"].items():
        model_select_info.append({
            "name": model_name,
            "paper_from": params["paper_from"],
            "year": params["year"]
        })
    data["pretrainedModelsInfo"] = model_select_info
    data["configLinks"] = {model_name: params["config_url"] for model_name, params in data["pretrainedModels"].items()}

    data["modelColumns"] = get_table_columns(metrics)

    state["selectedModel"] = {pretrained_model: data["pretrainedModels"][pretrained_model]["checkpoints"][0]['name']
                              for pretrained_model in data["pretrainedModels"].keys()}
    state["useAuxiliaryHead"] = True
    state["weightsInitialization"] = "pretrained"  # "custom"
    state["collapsed5"] = True
    state["disabled5"] = True
    state["weightsPath"] = ""
    data["done5"] = False
    state["loadingModel"] = False

    # default hyperparams that may be reassigned from model default params
    init_dc.init_default_cfg_params(state)

    ProgressBar(g.task_id, g.api, "data.progress6", "Download weights", is_size=True,
                                min_report_percent=5).init_data(data)


def get_pretrained_models(return_metrics=False):
    model_yamls = sly.json.load_json_file(os.path.join(g.root_source_dir, "models", "model_meta.json"))
    model_config = {}
    all_metrics = []
    for model_meta in model_yamls:
        with open(os.path.join(g.configs_dir, model_meta["yml_file"]), "r") as stream:
            model_info = yaml.safe_load(stream)
            model_config[model_meta["model_name"]] = {}
            model_config[model_meta["model_name"]]["checkpoints"] = []
            model_config[model_meta["model_name"]]["paper_from"] = model_meta["paper_from"]
            model_config[model_meta["model_name"]]["year"] = model_meta["year"]
            mmseg_ver = pkg_resources.get_distribution("mmsegmentation").version
            model_config[model_meta["model_name"]]["config_url"] = f"https://github.com/open-mmlab/mmsegmentation/tree/v{mmseg_ver}/configs/" + model_meta["yml_file"].split("/")[0]
            checkpoint_keys = []
            for model in model_info["Models"]:
                checkpoint_info = {}
                checkpoint_info["name"] = model["Name"]
                checkpoint_info["method"] = model["In Collection"]
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
        {"key": "method", "title": "Method", "subtitle": None},
        {"key": "dataset", "title": "Dataset", "subtitle": None},
        {"key": "inference_time", "title": "Inference time", "subtitle": "(ms/im)"},
        {"key": "crop_size", "title": "Input size", "subtitle": "(H, W)"},
        {"key": "lr_schd", "title": "LR scheduler", "subtitle": "steps"},
        {"key": "training_memory", "title": "Memory", "subtitle": "Training (GB)"},
    ]
    for metric in metrics:
        columns.append({"key": metric, "title": metric, "subtitle": "score"})
    return columns


def download_sly_file(remote_path, local_path, progress):
    file_info = g.api.file.get_info_by_path(g.team_id, remote_path)
    if file_info is None:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), remote_path)
    progress.set_total(file_info.sizeb)
    g.api.file.download(g.team_id, remote_path, local_path, g.my_app.cache,
                        progress.increment)
    progress.reset_and_update()

    sly.logger.info(f"{remote_path} has been successfully downloaded",
                    extra={"weights": local_path})


def download_custom_config(state):
    progress = ProgressBar(g.task_id, g.api, "data.progress6", "Download config", is_size=True,
                                           min_report_percent=5)

    weights_remote_dir = os.path.dirname(state["weightsPath"])
    g.model_config_local_path = os.path.join(g.checkpoints_dir, g.my_app.data_dir.split('/')[-1], 'custom_loaded_config.py')

    config_remote_dir = os.path.join(weights_remote_dir, f'config.py')
    if g.api.file.exists(g.team_id, config_remote_dir):
        download_sly_file(config_remote_dir, g.model_config_local_path, progress)


@g.my_app.callback("download_weights")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def download_weights(api: sly.Api, task_id, context, state, app_logger):
    progress = ProgressBar(g.task_id, g.api, "data.progress6", "Download weights", is_size=True,
                                           min_report_percent=5)
    try:
        if state["weightsInitialization"] == "custom":
            weights_path_remote = state["weightsPath"]
            if not weights_path_remote.endswith(".pth"):
                raise ValueError(f"Weights file has unsupported extension {sly.fs.get_file_ext(weights_path_remote)}. "
                                 f"Supported: '.pth'")

            g.local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_path_remote))
            if sly.fs.file_exists(g.local_weights_path):
                os.remove(g.local_weights_path)

            download_sly_file(weights_path_remote, g.local_weights_path, progress)
            download_custom_config(state)

        else:
            checkpoints_by_model = get_pretrained_models()[state["pretrainedModel"]]["checkpoints"]
            selected_model = next(item for item in checkpoints_by_model
                                  if item["name"] == state["selectedModel"][state["pretrainedModel"]])

            weights_url = selected_model.get('weights')
            config_file = selected_model.get('config_file')
            if weights_url is not None:
                g.local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_url))
                g.model_config_local_path = os.path.join(g.root_source_dir, config_file)
                if sly.fs.file_exists(g.local_weights_path) is False:
                    response = requests.head(weights_url, allow_redirects=True)
                    sizeb = int(response.headers.get('content-length', 0))
                    progress.set_total(sizeb)
                    os.makedirs(os.path.dirname(g.local_weights_path), exist_ok=True)
                    sly.fs.download(weights_url, g.local_weights_path, g.my_app.cache, progress.increment)
                    progress.reset_and_update()
                sly.logger.info("Pretrained weights has been successfully downloaded",
                                extra={"weights": g.local_weights_path})



    except Exception as e:
        progress.reset_and_update()
        raise e

    fields = [
        {"field": "state.loadingModel", "payload": False},
        {"field": "data.done5", "payload": True},
        {"field": "state.collapsed6", "payload": False},
        {"field": "state.disabled6", "payload": False},
        {"field": "state.activeStep", "payload": 6},
    ]

    global cfg
    cfg = Config.fromfile(g.model_config_local_path)
    if state["weightsInitialization"] != "custom":
        cfg.pretrained_model = state["pretrainedModel"]
    # print(f'Config:\n{cfg.pretty_text}') # TODO: debug
    params = init_dc.init_default_cfg_args(cfg)
    fields.extend(params)
    if not hasattr(cfg.model, "auxiliary_head") or cfg.model.auxiliary_head is None:
        fields.extend([
            {"field": "state.useAuxiliaryHead", "payload": False}
        ])

    g.api.app.set_fields(g.task_id, fields)

def restart(data, state):
    data["done5"] = False