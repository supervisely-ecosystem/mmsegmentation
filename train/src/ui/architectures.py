import errno
import os
import requests
import yaml
import pkg_resources
import sly_globals as g
import supervisely_lib as sly
from mmcv import Config

cfg = None

def init_default_cfg_params(state):
    state["optimizer"] = "SGD"
    state["lr"] = 0.001
    state["weightDecay"] = 0
    state["decodeHeadLoss"] = "CrossEntropyLoss"
    state["auxiliaryHeadLoss"] = "CrossEntropyLoss"
    state["decodeHeadLossWeight"] = 1.0
    state["auxiliaryHeadLossWeight"] = 0.4
    state["lrPolicy"] = "Cyclic"
    state["useWarmup"] = False
    state["warmup"] = "constant"
    state["warmupIters"] = 0
    state["warmupRatio"] = 0.1
    state["schedulerByEpochs"] = False
    state["minLREnabled"] = False
    state["minLR"] = None
    state["minLRRatio"] = None
    state["power"] = 1
    state["momentum"] = 0.9
    state["beta1"] = 0.9
    state["beta2"] = 0.999
    state["input_size"] = {
        "value": {
            "width": 256,
            "height": 256,
            "proportional": False
        },
        "options": {
            "proportions": {
              "width": 100,
              "height": 100
            },
            "min": 64
        }
    }
    state["batchSizePerGPU"] = 4
    state["workersPerGPU"] = 2

def init(data, state):
    state['pretrainedModel'] = 'SegFormer'
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
    init_default_cfg_params(state)

    sly.app.widgets.ProgressBar(g.task_id, g.api, "data.progress6", "Download weights", is_size=True,
                                min_report_percent=5).init_data(data)
    '''
    data["github_icon"] = {
        "imageUrl": "https://github.githubassets.com/favicons/favicon.png",
        "rounded": False,
        "bgColor": "rgba(0,0,0,0)"
    }
    data["arxiv_icon"] = {
        "imageUrl": "https://static.arxiv.org/static/browse/0.3.2.8/images/icons/favicon.ico",
        "rounded": False,
        "bgColor": "rgba(0,0,0,0)"
    }
    '''

def get_pretrained_models(return_metrics=False):
    model_yamls = sly.json.load_json_file(os.path.join(g.root_source_dir, "models", "model_meta.json"))
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
    progress = sly.app.widgets.ProgressBar(g.task_id, g.api, "data.progress6", "Download config", is_size=True,
                                           min_report_percent=5)

    weights_remote_dir = os.path.dirname(state["weightsPath"])
    g.model_config_local_path = os.path.join(g.checkpoints_dir, g.my_app.data_dir.split('/')[-1], 'custom_loaded_config.py')

    config_remote_dir = os.path.join(weights_remote_dir, f'config.py')
    if g.api.file.exists(g.team_id, config_remote_dir):
        download_sly_file(config_remote_dir, g.model_config_local_path, progress)


def init_default_cfg_args(cfg):
    params = [
        {
            "field": "state.decodeHeadLoss",
            "payload": cfg.model.decode_head[0].loss_decode.type if isinstance(cfg.model.decode_head, list) else cfg.model.decode_head.loss_decode.type
        },
        {
            "field": "state.decodeHeadLossWeight",
            "payload": cfg.model.decode_head[0].loss_decode.loss_weight if isinstance(cfg.model.decode_head, list) else cfg.model.decode_head.loss_decode.loss_weight
        },
    ]
    if hasattr(cfg.model, "auxiliary_head") and cfg.model.auxiliary_head is not None:
        params.extend([
            {
                "field": "state.auxiliaryHeadLoss",
                "payload": cfg.model.decode_head[0].loss_decode.type if isinstance(cfg.model.decode_head,
                                                                                list) else cfg.model.decode_head.loss_decode.type
            },
            {
                "field": "state.auxiliaryHeadLossWeight",
                "payload": cfg.model.auxiliary_head[0].loss_decode.loss_weight if isinstance(cfg.model.auxiliary_head, list) else cfg.model.auxiliary_head.loss_decode.loss_weight
            }
        ])
    if hasattr(cfg.data, "samples_per_gpu"):
        params.extend([{
            "field": "state.batchSizePerGPU",
            "payload": cfg.data.samples_per_gpu
        }])
    if hasattr(cfg.data, "workers_per_gpu"):
        params.extend([{
            "field": "state.workersPerGPU",
            "payload": cfg.data.workers_per_gpu
        }])
    if hasattr(cfg, "crop_size"):
        params.extend([{
            "field": "state.input_size.value.height",
            "payload": cfg.crop_size[0]
        },{
            "field": "state.input_size.value.width",
            "payload": cfg.crop_size[1]
        },{
            "field": "state.input_size.options.proportions.height",
            "payload": 100
        },{
            "field": "state.input_size.options.proportions.width",
            "payload": 100 * (cfg.crop_size[1] / cfg.crop_size[0])
        }])
    if hasattr(cfg.optimizer, "type"):
        params.extend([{
            "field": "state.optimizer",
            "payload": cfg.optimizer.type
        }])
    if hasattr(cfg.optimizer, "lr"):
        params.extend([{
            "field": "state.lr",
            "payload": cfg.optimizer.lr
        }])
    if hasattr(cfg.optimizer, "weight_decay"):
        params.extend([{
            "field": "state.weightDecay",
            "payload": cfg.optimizer.weight_decay
        }])
    if hasattr(cfg.optimizer, "momentum"):
        params.extend([{
            "field": "state.momentum",
            "payload": cfg.optimizer.momentum
        }])
    if hasattr(cfg.optimizer, "betas"):
        params.extend([{
            "field": "state.beta1",
            "payload": cfg.optimizer.betas[0]
        },{
            "field": "state.beta2",
            "payload": cfg.optimizer.betas[1]
        }])
    # take lr scheduler params
    if hasattr(cfg, "lr_config"):
        if hasattr(cfg.lr_config, "policy"):
            policy = cfg.lr_config.policy.capitalize()
            params.extend([{
                "field": "state.lrPolicy",
                "payload": "Cyclic"
            }])
        if hasattr(cfg.lr_config, "warmup"):
            params.extend([{
                "field": "state.useWarmup",
                "payload": True
            },{
                "field": "state.warmup",
                "payload": cfg.lr_config.warmup
            }])
        if hasattr(cfg.lr_config, "warmup_iters"):
            params.extend([{
                "field": "state.warmupIters",
                "payload": cfg.lr_config.warmup_iters
            }])
        if hasattr(cfg.lr_config, "warmup_ratio"):
            params.extend([{
                "field": "state.warmupRatio",
                "payload": cfg.lr_config.warmup_ratio
            }])
        if hasattr(cfg.lr_config, "by_epoch"):
            params.extend([{
                "field": "state.schedulerByEpochs",
                "payload": cfg.lr_config.by_epoch
            }])
        if hasattr(cfg.lr_config, "min_lr"):
            params.extend([{
                "field": "state.minLREnabled",
                "payload": True
            },{
                "field": "state.minLR",
                "payload": cfg.lr_config.min_lr
            }])
        if hasattr(cfg.lr_config, "power"):
            params.extend([{
                "field": "state.power",
                "payload": cfg.lr_config.power
            }])

    return params


@g.my_app.callback("download_weights")
@sly.timeit
# @g.my_app.ignore_errors_and_show_dialog_window()
def download_weights(api: sly.Api, task_id, context, state, app_logger):
    progress = sly.app.widgets.ProgressBar(g.task_id, g.api, "data.progress6", "Download weights", is_size=True,
                                           min_report_percent=5)
    try:
        if state["weightsInitialization"] == "custom":
            # raise NotImplementedError
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
                # TODO: check that pretrained weights are exist on remote server
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
    # print(f'Config:\n{cfg.pretty_text}')
    params = init_default_cfg_args(cfg)
    fields.extend(params)
    if cfg.pretrained_model in ["CGNet", "DPT", "ERFNet", "HRNet", "MobileNetV3", "OCRNet", "PointRend", "SegFormer", "SemanticFPN", "Twins"]:
        fields.extend([
            {"field": "state.useAuxiliaryHead", "payload": False}
        ])

    g.api.app.set_fields(g.task_id, fields)

def restart(data, state):
    data["done5"] = False