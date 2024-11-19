import supervisely as sly
from typing import Dict, Optional, Union
from sly_train_progress import init_progress, _update_progress_ui
import sly_globals as g
import os
import shutil
import cv2
import numpy as np
from functools import partial
from mmcv.cnn.utils import revert_sync_batchnorm
from mmseg.apis import train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from init_cfg import init_cfg
from sly_functions import get_bg_class_name
import workflow as w

# ! required to be left here despite not being used
import sly_imgaugs
import sly_dataset
import sly_logger_hook

_open_lnk_name = "open_app.lnk"


def init(data, state):
    init_progress("Epoch", data)
    init_progress("Iter", data)
    init_progress("UploadDir", data)
    data["eta"] = None
    state["isValidation"] = False

    # Device selctor
    data["multiple"] = False
    data["placeholder"] = "Select device"
    data["availableDevices"] = init_devices()
    state["deviceLoading"] = False
    if len(data["availableDevices"]) > 0:
        state["selectedDevice"] = data["availableDevices"][0]["value"]
    else:
        state["selectedDevice"] = None

    init_charts(data, state)

    state["collapsed7"] = True
    state["disabled7"] = True
    state["done7"] = False

    state["started"] = False
    state["preparingData"] = False
    data["outputName"] = None
    data["outputUrl"] = None


def init_devices():
    try:
        from torch import cuda
    except ImportError as ie:
        sly.logger.warn(
            "Unable to import Torch. Please, run 'pip install torch' to resolve the issue.",
            extra={"error message": str(ie)},
        )
        return

    devices = []
    cuda.init()
    if not cuda.is_available():
        sly.logger.warn("CUDA is not available")
        return

    for idx in range(cuda.device_count()):
        current_device = f"cuda:{idx}"
        full_device_name = f"{cuda.get_device_name(idx)} ({current_device})"
        free_mem, total_mem = cuda.mem_get_info(current_device)
        convert_to_gb = lambda number: round(number / 1024**3, 1)
        right_text = f"{convert_to_gb(total_mem - free_mem)} GB / {convert_to_gb(total_mem)} GB"
        device_info = {
            "value": idx,
            "label": full_device_name,
            "right_text": right_text,
            "free": free_mem,
        }
        devices.append(device_info)
    return devices


def init_chart(
    title, names, xs, ys, smoothing=None, yrange=None, decimals=None, xdecimals=None, metric=None
):
    series = []
    for name, x, y in zip(names, xs, ys):
        series.append({"name": name, "data": [[px, py] for px, py in zip(x, y)]})
    result = {"options": {"title": title}, "series": series}
    if len(names) > 0:
        result["series"] = series
    if metric is not None:
        result["metric"] = metric
    if smoothing is not None:
        result["options"]["smoothingWeight"] = smoothing
    if yrange is not None:
        result["options"]["yaxisInterval"] = yrange
    if decimals is not None:
        result["options"]["decimalsInFloat"] = decimals
    if xdecimals is not None:
        result["options"]["xaxisDecimalsInFloat"] = xdecimals
    return result


def init_charts(data, state):
    state["smoothing"] = 0.6
    state["chartLR"] = init_chart(
        "LR",
        names=["lr"],
        xs=[[]],
        ys=[[]],
        smoothing=None,
        # yrange=[state["lr"] - state["lr"] / 2.0, state["lr"] + state["lr"] / 2.0],
        decimals=6,
        xdecimals=2,
    )
    state["chartTrainLoss"] = init_chart(
        "Train Loss",
        names=["loss"],
        xs=[[]],
        ys=[[]],
        smoothing=state["smoothing"],
        decimals=6,
        xdecimals=2,
    )
    state["mean_charts"] = {}
    for metric in data["availableMetrics"]:
        state["mean_charts"][f"chartVal_{metric}"] = init_chart(
            f"Val {metric}",
            metric=metric,
            names=[metric],
            xs=[[]],
            ys=[[]],
            smoothing=state["smoothing"],
        )
    state["class_charts"] = {}
    for metric in data["availableMetrics"]:
        state["class_charts"][f"chartVal_{metric[1:]}"] = init_chart(
            f"Val {metric[1:]}", names=[], metric=metric, xs=[], ys=[], smoothing=state["smoothing"]
        )

    state["chartTime"] = init_chart("Time", names=["time"], xs=[[]], ys=[[]], xdecimals=2)
    state["chartDataTime"] = init_chart(
        "Data Time", names=["data_time"], xs=[[]], ys=[[]], xdecimals=2
    )
    state["chartMemory"] = init_chart("Memory", names=["memory"], xs=[[]], ys=[[]], xdecimals=2)


@g.my_app.callback("refresh_devices")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def refresh_devices(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        {"field": "data.availableDevices", "payload": init_devices()},
        {"field": "state.selectedDevice", "payload": state["selectedDevice"]},
        {"field": "state.deviceLoading", "payload": False},
    ]
    g.api.app.set_fields(g.task_id, fields)


@g.my_app.callback("change_smoothing")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def change_smoothing(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        {"field": "state.chartTrainLoss.options.smoothingWeight", "payload": state["smoothing"]}
    ]
    for metric in state["evalMetrics"]:
        fields.extend(
            [
                {
                    "field": f"state.mean_charts.chartVal_{metric}.options.smoothingWeight",
                    "payload": state["smoothing"],
                },
                {
                    "field": f"state.class_charts.chartVal_{metric[1:]}.options.smoothingWeight",
                    "payload": state["smoothing"],
                },
            ]
        )
    g.api.app.set_fields(g.task_id, fields)


def _save_link_to_ui(local_dir, app_url):
    # save report to file *.lnk (link to report)
    local_path = os.path.join(local_dir, _open_lnk_name)
    sly.fs.ensure_base_path(local_path)
    with open(local_path, "w") as text_file:
        print(app_url, file=text_file)


def upload_artifacts_and_log_progress():
    _save_link_to_ui(g.artifacts_dir, g.my_app.app_url)

    current_progress = 0
    last_read = 0

    def upload_monitor(monitor, api: sly.Api, task_id, progress: sly.tqdm_sly):
        nonlocal last_read, current_progress, dir_size

        if monitor.bytes_read < last_read:
            last_read = 0
        elif 0 < monitor.bytes_read < 1024 * 16: # if next batch is less than 16 KB
            last_read = 0
        diff = monitor.bytes_read - last_read
        last_read = monitor.bytes_read
        current_progress += diff
        if progress.total == 0:
            progress.set(current_progress, dir_size, report=False)
        else:
            progress.set_current_value(current_progress, report=False)
        _update_progress_ui("UploadDir", g.api, g.task_id, progress)

    dir_size = sly.fs.get_directory_size(g.artifacts_dir)
    progress = sly.Progress(
        "Upload directory with training artifacts to Team Files", dir_size, is_size=True
    )
    progress_cb = partial(upload_monitor, api=g.api, task_id=g.task_id, progress=progress)

    model_dir = g.sly_mmseg.framework_folder
    remote_artifacts_dir = f"{model_dir}/{g.task_id}_{g.project_info.name}"
    remote_weights_dir = os.path.join(remote_artifacts_dir, g.sly_mmseg.weights_folder)
    remote_config_path = os.path.join(remote_weights_dir, g.sly_mmseg.config_file)

    res_dir = g.api.file.upload_directory(
        g.team_id, g.artifacts_dir, remote_artifacts_dir, progress_size_cb=progress_cb
    )

    # generate metadata file
    g.sly_mmseg_generated_metadata = g.sly_mmseg.generate_metadata(
        app_name=g.sly_mmseg.app_name,
        task_id=g.task_id,
        artifacts_folder=remote_artifacts_dir,
        weights_folder=remote_weights_dir,
        weights_ext=g.sly_mmseg.weights_ext,
        project_name=g.project_info.name,
        task_type=g.sly_mmseg.task_type,
        config_path=remote_config_path,
    )

    return res_dir


def init_class_charts_series(state):
    classes = state["selectedClasses"]
    bg = get_bg_class_name(classes)
    if bg is None:
        classes = classes + ["__bg__"]
    series = []
    for class_name in classes:
        series.append({"name": class_name, "data": []})
    fields = [{"field": "state.preparingData", "payload": True}]
    for metric_name in state["evalMetrics"]:
        fields.extend(
            [{"field": f"state.class_charts.chartVal_{metric_name[1:]}.series", "payload": series}]
        )
    g.api.app.set_fields(g.task_id, fields)


def prepare_segmentation_data(state, img_dir, ann_dir, palette, target_classes=None):
    target_classes = target_classes or state["selectedClasses"]
    temp_project_seg_dir = g.project_seg_dir + "_temp"
    sly.Project.to_segmentation_task(
        g.project_dir, temp_project_seg_dir, target_classes=target_classes
    )

    datasets = os.listdir(temp_project_seg_dir)
    os.makedirs(os.path.join(g.project_seg_dir, img_dir), exist_ok=True)
    os.makedirs(os.path.join(g.project_seg_dir, ann_dir), exist_ok=True)
    for dataset in datasets:
        if not os.path.isdir(os.path.join(temp_project_seg_dir, dataset)):
            if dataset == "meta.json":
                shutil.move(os.path.join(temp_project_seg_dir, "meta.json"), g.project_seg_dir)
            continue
        # convert masks to required format and save to general ann_dir
        mask_files = os.listdir(os.path.join(temp_project_seg_dir, dataset, ann_dir))
        for mask_file in mask_files:
            mask = cv2.imread(os.path.join(temp_project_seg_dir, dataset, ann_dir, mask_file))[
                :, :, ::-1
            ]
            result = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int32)
            # human masks to machine masks
            for color_idx, color in enumerate(palette):
                colormap = np.where(np.all(mask == color, axis=-1))
                result[colormap] = color_idx
            cv2.imwrite(os.path.join(g.project_seg_dir, ann_dir, mask_file), result)

        imgfiles_to_move = os.listdir(os.path.join(temp_project_seg_dir, dataset, img_dir))
        for filename in imgfiles_to_move:
            shutil.move(
                os.path.join(temp_project_seg_dir, dataset, img_dir, filename),
                os.path.join(g.project_seg_dir, img_dir),
            )

    shutil.rmtree(temp_project_seg_dir)
    g.api.app.set_field(g.task_id, "state.preparingData", False)


@g.my_app.callback("train")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def train(api: sly.Api, task_id, context, state, app_logger):
    init_class_charts_series(state)
    try:
        sly.json.dump_json_file(state, os.path.join(g.info_dir, "ui_state.json"))

        img_dir = "img"
        ann_dir = "seg"
        obj_classes = g.project_meta.obj_classes
        cls_names = [obj_class.name for obj_class in obj_classes]
        bg_name = get_bg_class_name(cls_names) or "__bg__"
        if g.project_meta.get_obj_class(bg_name) is None:
            obj_classes = obj_classes.add(
                sly.ObjClass(name=bg_name, geometry_type=sly.Bitmap, color=(0, 0, 0))
            )
        classes_json = obj_classes.to_json()
        classes_json = [
            obj
            for obj in classes_json
            if obj["title"] in state["selectedClasses"] or obj["title"] == bg_name
        ]
        classes = [obj["title"] for obj in classes_json]
        palette = [obj["color"].lstrip("#") for obj in classes_json]
        palette = [[int(color[i : i + 2], 16) for i in (0, 2, 4)] for color in palette]
        if not os.path.exists(g.project_seg_dir):
            prepare_segmentation_data(state, img_dir, ann_dir, palette, classes)

        cfg = init_cfg(state, img_dir, ann_dir, classes, palette)
        # print(f'Config:\n{cfg.pretty_text}') # TODO: debug
        os.makedirs(os.path.join(g.checkpoints_dir, cfg.work_dir.split("/")[-1]), exist_ok=True)
        cfg.dump(os.path.join(g.checkpoints_dir, cfg.work_dir.split("/")[-1], "config.py"))

        # Build the dataset
        datasets = [build_dataset(cfg.data.train)]

        # Build the detector
        model = build_segmentor(
            cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
        )
        # Add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES
        model = revert_sync_batchnorm(model)
        # Create work_dir
        os.makedirs(os.path.abspath(cfg.work_dir), exist_ok=True)
        train_segmentor(model, datasets, cfg, distributed=False, validate=True, meta=dict())

        # hide progress bars and eta
        fields = [
            {"field": "data.progressEpoch", "payload": None},
            {"field": "data.progressIter", "payload": None},
            {"field": "data.eta", "payload": None},
        ]
        g.api.app.set_fields(g.task_id, fields)

        remote_dir = upload_artifacts_and_log_progress()
        file_info = api.file.get_info_by_path(g.team_id, os.path.join(remote_dir, _open_lnk_name))
        api.task.set_output_directory(task_id, file_info.id, remote_dir)

        fields = [
            {"field": "data.outputUrl", "payload": g.api.file.get_url(file_info.id)},
            {"field": "data.outputName", "payload": remote_dir},
            {"field": "state.done7", "payload": True},
            {"field": "state.started", "payload": False},
        ]
        g.api.app.set_fields(g.task_id, fields)

        w.workflow_input(api, g.project_info, state)
        w.workflow_output(api, g.sly_mmseg_generated_metadata, state)

        # stop application
        g.my_app.stop()
    except Exception as e:
        g.api.app.set_field(task_id, "state.started", False)
        raise e  # app will handle this error and show modal window
