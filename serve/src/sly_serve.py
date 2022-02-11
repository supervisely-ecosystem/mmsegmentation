import supervisely as sly
import functools
import sly_globals as g
import utils
import os

def send_error_data(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = None
        try:
            value = func(*args, **kwargs)
        except Exception as e:
            request_id = kwargs["context"]["request_id"]
            g.my_app.send_response(request_id, data={"error": repr(e)})
        return value
    return wrapper

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
        "device": g.device,
        "session_id": task_id,
        "classes_count": len(g.meta.obj_classes),
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
    results = utils.inference_image_path(local_image_path, context, state, app_logger)
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
    ann_json = utils.inference_image_path(image_path, context, state, app_logger)
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

    results = []
    for image_path in paths:
        ann_json = utils.inference_image_path(image_path, context, state, app_logger)
        results.append(ann_json)
        sly.fs.silent_remove(image_path)

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=results)


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
    sly.logger.info("Model has been successfully deployed")


def init_state_and_data(data, state):
    state['pretrainedModel'] = 'SegFormer'
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
    state["device"] = "cuda:0"
    state["weightsPath"] = ""
    state["loading"] = False
    state["deployed"] = False
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