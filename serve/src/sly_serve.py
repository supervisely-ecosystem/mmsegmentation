import traceback
import torch
import supervisely as sly
import functools
import sly_globals as g
import utils
import os
import sly_apply_nn_to_video as nn_to_video


def send_error_data(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = None
        try:
            value = func(*args, **kwargs)
        except Exception as e:
            sly.logger.error(f"Error while processing data: {e}")
            request_id = kwargs["context"]["request_id"]
            # raise e
            try:
                g.my_app.send_response(request_id, data={"error": repr(e)})
                print(traceback.format_exc())
            except Exception as ex:
                sly.logger.exception(f"Cannot send error response: {ex}")
        return value

    return wrapper


def inference_images_dir(img_paths, context, state, app_logger):
    annotations = []
    for image_path in img_paths:
        ann_json = utils.inference_image_path(image_path=image_path,
                                              project_meta=g.meta,
                                              context=context,
                                              state=state,
                                              app_logger=app_logger)
        annotations.append(ann_json)
        sly.fs.silent_remove(image_path)
    return annotations


@g.my_app.callback("get_output_classes_and_tags")
@sly.timeit
def get_output_classes_and_tags(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=g.meta.to_json())


@g.my_app.callback("get_custom_inference_settings")
@sly.timeit
def get_custom_inference_settings(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data={"settings": {}})


@g.my_app.callback("get_session_info")
@sly.timeit
@send_error_data
def get_session_info(api: sly.Api, task_id, context, state, app_logger):
    info = {
        "app": "MM Segmentation Serve",
        "type": "Semantic Segmentation",
        "device": g.device,
        "session_id": task_id,
        "classes_count": len(g.meta.obj_classes),
        "videos_support": True
    }
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=info)


@g.my_app.callback("inference_image_url")
@sly.timeit
@send_error_data
def inference_image_url(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    image_url = state["image_url"]
    ext = sly.fs.get_file_ext(image_url)
    if ext == "":
        ext = ".jpg"
    local_image_path = os.path.join(g.my_app.data_dir, sly.rand_str(15) + ext)
    sly.fs.download(image_url, local_image_path)
    results = utils.inference_image_path(image_path=local_image_path, project_meta=g.meta,
                                         context=context, state=state, app_logger=app_logger)
    sly.fs.silent_remove(local_image_path)

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=results)


@g.my_app.callback("inference_image_id")
@sly.timeit
def inference_image_id(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    image_id = state["image_id"]
    image_info = api.image.get_info_by_id(image_id)
    image_path = os.path.join(g.my_app.data_dir, sly.rand_str(10) + image_info.name)
    api.image.download_path(image_id, image_path)
    ann_json = utils.inference_image_path(image_path=image_path, project_meta=g.meta,
                                          context=context, state=state, app_logger=app_logger)
    sly.fs.silent_remove(image_path)
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=ann_json)


@g.my_app.callback("inference_batch_ids")
@sly.timeit
def inference_batch_ids(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    ids = state["batch_ids"]
    infos = api.image.get_info_by_id_batch(ids)
    paths = []
    for info in infos:
        paths.append(os.path.join(g.my_app.data_dir, sly.rand_str(10) + info.name))
    api.image.download_paths(infos[0].dataset_id, ids, paths)

    results = inference_images_dir(paths, context, state, app_logger)

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=results)


@g.my_app.callback("inference_video_id")
@sly.timeit
@send_error_data
def inference_video_id(api: sly.Api, task_id, context, state, app_logger):
    video_info = g.api.video.get_info_by_id(state['videoId'])

    sly.logger.info(f'inference {video_info.id=} started')
    inf_video_interface = nn_to_video.InferenceVideoInterface(api=g.api,
                                                              start_frame_index=state.get('startFrameIndex', 0),
                                                              frames_count=state.get('framesCount',
                                                                                     video_info.frames_count - 1),
                                                              frames_direction=state.get('framesDirection', 'forward'),
                                                              video_info=video_info,
                                                              imgs_dir=os.path.join(g.my_app.data_dir,
                                                                                    'videoInference'))

    inf_video_interface.download_frames()

    annotations = inference_images_dir(img_paths=inf_video_interface.images_paths,
                                       context=context,
                                       state=state,
                                       app_logger=app_logger)

    g.my_app.send_response(context["request_id"], data={'ann': annotations})
    sly.logger.info(f'inference {video_info.id=} done, {len(annotations)} annotations created')


@g.my_app.callback("run")
@g.my_app.ignore_errors_and_show_dialog_window()
def init_model(api: sly.Api, task_id, context, state, app_logger):
    g.remote_weights_path = state["weightsPath"]
    g.device = state["device"]
    utils.download_weights(state)
    utils.init_model_and_cfg(state)
    fields = [
        {"field": "state.loading", "payload": False},
        {"field": "state.deployed", "payload": True},
    ]
    g.api.app.set_fields(g.TASK_ID, fields)
    sly.logger.info("🟩 Model has been successfully deployed")


def init_state_and_data(data, state):
    state['pretrainedModel'] = 'ConvNeXt'
    data["pretrainedModels"], metrics = utils.get_pretrained_models(return_metrics=True)
    model_select_info = []
    for model_name, params in data["pretrainedModels"].items():
        model_select_info.append({
            "name": model_name,
            "paper_from": params["paper_from"],
            "year": params["year"]
        })
    data["pretrainedModelsInfo"] = model_select_info
    data["configLinks"] = {model_name: params["config_url"] for model_name, params in data["pretrainedModels"].items()}

    data["modelColumns"] = utils.get_table_columns(metrics)
    state["weightsInitialization"] = "pretrained"
    state["selectedModel"] = {pretrained_model: data["pretrainedModels"][pretrained_model]["checkpoints"][0]['name']
                              for pretrained_model in data["pretrainedModels"].keys()}
    device_values = ["cpu"]
    device_names = ["CPU"]
    if torch.cuda.is_available():
        gpus = torch.cuda.device_count()
        for i in range(gpus):
            device_values.append(f"cuda:{i}")
            device_names.append(f"{torch.cuda.get_device_name(i)} (cuda:{i})")

    data["available_device_names"] = device_names
    data["available_device_values"] = device_values
    state["device"] = device_values[0]
    state["weightsPath"] = ""
    state["loading"] = False
    state["deployed"] = False


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.TEAM_ID,
        "context.workspaceId": g.WORKSPACE_ID
    })
    data = {}
    state = {}

    init_state_and_data(data, state)

    g.my_app.compile_template(g.root_source_path)
    g.my_app.run(data=data, state=state)


if __name__ == "__main__":
    sly.main_wrapper("main", main)
