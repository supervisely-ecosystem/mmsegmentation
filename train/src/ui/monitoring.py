import supervisely as sly
from sly_train_progress import init_progress, _update_progress_ui
import sly_globals as g
import os
import shutil
import cv2
import math
import numpy as np
from functools import partial
from mmcv.cnn.utils import revert_sync_batchnorm
from mmseg.apis import train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from init_cfg import init_cfg
from sly_functions import get_bg_class_name, get_eval_results_dir_name
from splits import get_train_val_sets
from supervisely.nn.inference import SessionJSON
from supervisely.nn.artifacts.artifacts import TrainInfo
from supervisely.io.json import dump_json_file
from dataclasses import asdict
import workflow as w

# ! required to be left here despite not being used
import sly_imgaugs
import sly_dataset
import sly_logger_hook


def external_update_callback(progress: sly.tqdm_sly, progress_name: str):
    percent = math.floor(progress.n / progress.total * 100)
    fields = []
    if hasattr(progress, "desc"):
        fields.append({"field": f"data.progress{progress_name}", "payload": progress.desc})
    elif hasattr(progress, "message"):
        fields.append({"field": f"data.progress{progress_name}", "payload": progress.message})
    fields += [
        {"field": f"data.progressCurrent{progress_name}", "payload": progress.n},
        {"field": f"data.progressTotal{progress_name}", "payload": progress.total},
        {"field": f"data.progressPercent{progress_name}", "payload": percent},
    ]
    g.api.app.set_fields(g.task_id, fields)


def external_close_callback(progress: sly.tqdm_sly, progress_name: str):
    fields = [
        {"field": f"data.progress{progress_name}", "payload": None},
        {"field": f"data.progressCurrent{progress_name}", "payload": None},
        {"field": f"data.progressTotal{progress_name}", "payload": None},
        {"field": f"data.progressPercent{progress_name}", "payload": None},
    ]
    g.api.app.set_fields(g.task_id, fields)


class TqdmBenchmark(sly.tqdm_sly):
    def update(self, n=1):
        super().update(n)
        external_update_callback(self, "Benchmark")

    def close(self):
        super().close()
        external_close_callback(self, "Benchmark")


class TqdmProgress(sly.tqdm_sly):
    def update(self, n=1):
        super().update(n)
        external_update_callback(self, "Tqdm")

    def close(self):
        super().close()
        external_close_callback(self, "Tqdm")


_open_lnk_name = "open_app.lnk"
m = None


def init(data, state):
    init_progress("Epoch", data)
    init_progress("Iter", data)
    init_progress("UploadDir", data)
    init_progress("Benchmark", data)
    init_progress("Tqdm", data)
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
    data["benchmarkUrl"] = None
    state["benchmarkInProgress"] = False


def init_devices():
    try:
        from torch import cuda
    except ImportError as ie:
        sly.logger.warning(
            "Unable to import Torch. Please, run 'pip install torch' to resolve the issue.",
            extra={"error message": str(ie)},
        )
        return

    devices = []
    cuda.init()
    if not cuda.is_available():
        sly.logger.warning("CUDA is not available")
        return devices

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
    return sorted(devices, key=lambda x: x["free"], reverse=True)


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
        elif 0 < monitor.bytes_read < 1024 * 16:  # if next batch is less than 16 KB
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


def prepare_segmentation_data(state, img_dir, ann_dir, palette, target_classes):
    target_classes = target_classes or state["selectedClasses"]
    temp_project_seg_dir = g.project_seg_dir + "_temp"
    bg_name = get_bg_class_name(target_classes) or "__bg__"
    bg_color = (0, 0, 0)
    if bg_name in target_classes:
        try:
            bg_color = palette[target_classes.index(bg_name)]
        except:
            pass

    project = sly.Project(g.project_dir, sly.OpenMode.READ)
    with TqdmProgress(
        message="Converting project to segmentation task",
        total=project.total_items,
    ) as p:
        sly.Project.to_segmentation_task(
            g.project_dir,
            temp_project_seg_dir,
            target_classes=target_classes,
            progress_cb=p.update,
            bg_color=bg_color,
            bg_name=bg_name,
        )

    palette_lookup = np.zeros(256**3, dtype=np.int32)
    for idx, color in enumerate(palette):
        key = (color[0] << 16) | (color[1] << 8) | color[2]
        palette_lookup[key] = idx

    temp_project_fs = sly.Project(temp_project_seg_dir, sly.OpenMode.READ)
    os.makedirs(os.path.join(g.project_seg_dir, img_dir), exist_ok=True)
    os.makedirs(os.path.join(g.project_seg_dir, ann_dir), exist_ok=True)
    total_items = temp_project_fs.total_items

    meta_path = os.path.join(g.project_seg_dir, "meta.json")
    sly.json.dump_json_file(temp_project_fs.meta.to_json(), meta_path)

    with TqdmProgress(
        message="Converting masks to required format",
        total=total_items,
    ) as p:
        for dataset in temp_project_fs.datasets:
            dataset: sly.Dataset
            if not sly.fs.dir_exists(dataset.seg_dir):
                continue
            # convert masks to required format and save to general ann_dir
            names = dataset.get_items_names()
            mask_files = [dataset.get_seg_path(name) for name in names]
            for path in mask_files:
                file_name = os.path.basename(path)
                mask = cv2.imread(path)[:, :, ::-1]
                result = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int32)
                # human masks to machine masks
                mask_keys = (
                    (mask[:, :, 0].astype(np.int32) << 16)
                    | (mask[:, :, 1].astype(np.int32) << 8)
                    | mask[:, :, 2].astype(np.int32)
                )
                result = palette_lookup[mask_keys]

                cv2.imwrite(os.path.join(g.project_seg_dir, ann_dir, file_name), result)
                p.update(1)

            imgfiles_to_move = [dataset.get_img_path(name) for name in names]

            for path in imgfiles_to_move:
                shutil.move(path, os.path.join(g.project_seg_dir, img_dir))

    shutil.rmtree(temp_project_seg_dir)
    g.api.app.set_field(g.task_id, "state.preparingData", False)


def run_benchmark(api: sly.Api, task_id, classes, cfg, state, remote_dir):
    global m

    api.app.set_field(task_id, "state.benchmarkInProgress", True)
    benchmark_report_template, report_id, eval_metrics, primary_metric_name = None, None, None, None
    try:
        from sly_mmsegm import MMSegmentationModelBench
        import torch
        from pathlib import Path
        import asyncio

        dataset_infos = api.dataset.get_list(g.project_id, recursive=True)

        dummy_pbar = TqdmProgress
        with dummy_pbar(message="Preparing trained model for benchmark", total=1) as p:
            # 0. Find the best checkpoint
            best_filename = None
            best_checkpoints = []
            latest_checkpoint = None
            other_checkpoints = []
            for root, dirs, files in os.walk(g.checkpoints_dir):
                for file_name in files:
                    path = os.path.join(root, file_name)
                    if file_name.endswith(".pth"):
                        if file_name.startswith("best_"):
                            best_checkpoints.append(path)
                        elif file_name == "latest.pth":
                            latest_checkpoint = path
                        elif file_name.startswith("epoch_"):
                            other_checkpoints.append(path)

            if len(best_checkpoints) > 1:
                best_checkpoints = sorted(best_checkpoints, key=lambda x: x, reverse=True)
            elif len(best_checkpoints) == 0:
                sly.logger.info("Best model checkpoint not found in the checkpoints directory.")
                if latest_checkpoint is not None:
                    best_checkpoints = [latest_checkpoint]
                    sly.logger.info(
                        f"Using latest checkpoint for evaluation: {latest_checkpoint!r}"
                    )
                elif len(other_checkpoints) > 0:
                    parse_epoch = lambda x: int(x.split("_")[-1].split(".")[0])
                    best_checkpoints = sorted(other_checkpoints, key=parse_epoch, reverse=True)
                    sly.logger.info(
                        f"Using the last epoch checkpoint for evaluation: {best_checkpoints[0]!r}"
                    )

            if len(best_checkpoints) == 0:
                raise ValueError("No checkpoints found for evaluation.")
            best_checkpoint = Path(best_checkpoints[0])
            sly.logger.info(f"Starting model benchmark with the checkpoint: {best_checkpoint!r}")
            best_filename = best_checkpoint.name
            workdir = best_checkpoint.parent

            # 1. Serve trained model
            m = MMSegmentationModelBench(model_dir=str(workdir), use_gui=False)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            sly.logger.info(f"Using device: {device}")

            checkpoint_path = g.sly_mmseg.get_weights_path(remote_dir) + "/" + best_filename
            config_path = g.sly_mmseg.get_config_path(remote_dir)
            sly.logger.info(f"Checkpoint path: {checkpoint_path}")

            try:
                arch_type = cfg.model.backbone.type
            except Exception as e:
                arch_type = "unknown"

            sly.logger.info(f"Model architecture: {arch_type}")

            deploy_params = dict(
                device=device,
                model_source="Custom models",
                task_type=sly.nn.TaskType.SEMANTIC_SEGMENTATION,
                checkpoint_name=best_filename,
                checkpoint_url=checkpoint_path,
                config_url=config_path,
                arch_type=arch_type,
            )
            m._load_model(deploy_params)
            asyncio.set_event_loop(asyncio.new_event_loop())
            m.serve()

            import requests
            import uvicorn
            import time
            from threading import Thread

            def run_app():
                uvicorn.run(m.app, host="localhost", port=8000)

            thread = Thread(target=run_app, daemon=True)
            thread.start()

            while True:
                try:
                    requests.get("http://localhost:8000")
                    print("✅ Local server is ready")
                    break
                except requests.exceptions.ConnectionError:
                    print("Waiting for the server to be ready")
                    time.sleep(0.1)

            session = SessionJSON(api, session_url="http://localhost:8000")
            if sly.fs.dir_exists(g.data_dir + "/benchmark"):
                sly.fs.remove_dir(g.data_dir + "/benchmark")

            # 1. Init benchmark (todo: auto-detect task type)
            benchmark_dataset_ids = None
            benchmark_images_ids = None
            train_dataset_ids = None
            train_images_ids = None

            split_method = state["splitMethod"]

            if split_method == "datasets":
                train_datasets = state["trainDatasets"]
                val_datasets = state["valDatasets"]
                benchmark_dataset_ids = [ds.id for ds in dataset_infos if ds.name in val_datasets]
                train_dataset_ids = [ds.id for ds in dataset_infos if ds.name in train_datasets]
                train_set, val_set = get_train_val_sets(g.project_dir, state)
            else:

                def get_image_infos_by_split(split: list):
                    ds_infos_dict = {ds_info.name: ds_info for ds_info in dataset_infos}
                    image_names_per_dataset = {}
                    for item in split:
                        name = item.name
                        if name[1] == "_":
                            name = name[2:]
                        elif name[2] == "_":
                            name = name[3:]
                        image_names_per_dataset.setdefault(item.dataset_name, []).append(name)
                    image_infos = []
                    for dataset_name, image_names in image_names_per_dataset.items():
                        if "/" in dataset_name:
                            dataset_name = dataset_name.split("/")[-1]
                        ds_info = ds_infos_dict[dataset_name]
                        for batched_names in sly.batched(image_names, 200):
                            batch_infos = api.image.get_list(
                                ds_info.id,
                                filters=[
                                    {
                                        "field": "name",
                                        "operator": "in",
                                        "value": batched_names,
                                    }
                                ],
                            )
                            image_infos.extend(batch_infos)
                    return image_infos

                train_set, val_set = get_train_val_sets(g.project_dir, state)

                val_image_infos = get_image_infos_by_split(val_set)
                train_image_infos = get_image_infos_by_split(train_set)
                benchmark_images_ids = [img_info.id for img_info in val_image_infos]
                train_images_ids = [img_info.id for img_info in train_image_infos]

            p.update(1)

        pbar = TqdmBenchmark
        bm = sly.nn.benchmark.SemanticSegmentationBenchmark(
            api,
            g.project_info.id,
            output_dir=g.data_dir + "/benchmark",
            gt_dataset_ids=benchmark_dataset_ids,
            gt_images_ids=benchmark_images_ids,
            progress=pbar,
            progress_secondary=pbar,
            classes_whitelist=classes,
        )

        train_info = {
            "app_session_id": sly.env.task_id(),
            "train_dataset_ids": train_dataset_ids,
            "train_images_ids": train_images_ids,
            "images_count": len(train_set),
        }
        bm.train_info = train_info

        # 2. Run inference
        bm.run_inference(session)

        # 3. Pull results from the server
        gt_project_path, pred_project_path = bm.download_projects(save_images=False)

        # 4. Evaluate
        bm._evaluate(gt_project_path, pred_project_path)
        bm._dump_eval_inference_info(bm._eval_inference_info)

        # 5. Upload evaluation results
        eval_res_dir = get_eval_results_dir_name(api, sly.env.task_id(), g.project_info)
        bm.upload_eval_results(eval_res_dir + "/evaluation/")

        # # 6. Speed test
        if state["runSpeedTest"]:
            try:
                session_info = session.get_session_info()
                support_batch_inference = session_info.get("batch_inference_support", False)
                max_batch_size = session_info.get("max_batch_size")
                batch_sizes = (1, 8, 16)
                if not support_batch_inference:
                    batch_sizes = (1,)
                elif max_batch_size is not None:
                    batch_sizes = tuple([bs for bs in batch_sizes if bs <= max_batch_size])
                bm.run_speedtest(session, g.project_info.id, batch_sizes=batch_sizes)
                bm.upload_speedtest_results(eval_res_dir + "/speedtest/")
            except Exception as e:
                sly.logger.warning(f"Speedtest failed. Skipping. {e}")

        # 7. Prepare visualizations, report and
        bm.visualize()
        remote_dir = bm.upload_visualizations(eval_res_dir + "/visualizations/")
        report_id = bm.report.id
        eval_metrics = bm.key_metrics
        primary_metric_name = bm.primary_metric_name

        # 8. UI updates
        benchmark_report_template = bm.report

        fields = [
            {"field": f"state.benchmarkInProgress", "payload": False},
            {"field": f"data.benchmarkUrl", "payload": bm.get_report_link()},
        ]
        api.app.set_fields(g.task_id, fields)
        sly.logger.info(
            f"Predictions project name: {bm.dt_project_info.name}. Workspace_id: {bm.dt_project_info.workspace_id}"
        )

        # 9. Stop the server
        try:
            m.app.stop()
        except Exception as e:
            sly.logger.warning(f"Failed to stop the model app: {e}")
        try:
            thread.join()
        except Exception as e:
            sly.logger.warning(f"Failed to stop the server: {e}")
    except Exception as e:
        sly.logger.error(f"Model benchmark failed. {repr(e)}", exc_info=True)
        try:
            if bm.dt_project_info:
                api.project.remove(bm.dt_project_info.id)
        except Exception as re:
            pass

    return benchmark_report_template, report_id, eval_metrics, primary_metric_name


def create_experiment(
    model_name, remote_dir, report_id=None, eval_metrics=None, primary_metric_name=None
):
    train_info = TrainInfo(**g.sly_mmseg_generated_metadata)
    experiment_info = g.sly_mmseg.convert_train_to_experiment_info(train_info)
    experiment_info.experiment_name = f"{g.task_id}_{g.project_info.name}_{model_name}"
    experiment_info.model_name = model_name
    experiment_info.framework_name = f"{g.sly_mmseg.framework_name}"
    experiment_info.train_size = g.train_size
    experiment_info.val_size = g.val_size
    experiment_info.evaluation_report_id = report_id
    if report_id is not None:
        experiment_info.evaluation_report_link = f"/model-benchmark?id={str(report_id)}"
    experiment_info.evaluation_metrics = eval_metrics

    experiment_info_json = asdict(experiment_info)
    experiment_info_json["project_preview"] = g.project_info.image_preview_url
    experiment_info_json["primary_metric"] = primary_metric_name

    g.api.task.set_output_experiment(g.task_id, experiment_info_json)
    experiment_info_json.pop("project_preview")
    experiment_info_json.pop("primary_metric")

    experiment_info_path = os.path.join(g.artifacts_dir, "experiment_info.json")
    remote_experiment_info_path = os.path.join(remote_dir, "experiment_info.json")
    dump_json_file(experiment_info_json, experiment_info_path)
    g.api.file.upload(g.team_id, experiment_info_path, remote_experiment_info_path)


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

        if state["saveLast"] is False:
            for root, _, files in os.walk(g.checkpoints_dir):
                for file in files:
                    if file == "latest.pth":
                        sly.fs.silent_remove(os.path.join(root, file))

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

        benchmark_report_template, report_id, eval_metrics, primary_metric_name = (
            None,
            None,
            None,
            None,
        )
        # run benchmark
        sly.logger.info(f"Run benchmark: {state['runBenchmark']}")
        if state["runBenchmark"]:
            benchmark_report_template, report_id, eval_metrics, primary_metric_name = run_benchmark(
                api, task_id, classes, cfg, state, remote_dir
            )

        sly.logger.info("Creating experiment info")
        create_experiment(
            state["pretrainedModel"], remote_dir, report_id, eval_metrics, primary_metric_name
        )
        w.workflow_input(api, g.project_info, state)
        w.workflow_output(api, g.sly_mmseg_generated_metadata, state, benchmark_report_template)

        # stop application
        g.my_app.stop()
    except Exception as e:
        g.api.app.set_field(task_id, "state.started", False)
        raise e  # app will handle this error and show modal window
