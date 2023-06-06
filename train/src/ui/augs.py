import os
import supervisely as sly
import sly_globals as g
from supervisely.app.v1.widgets.compare_gallery import CompareGallery
import input_project


_templates = [
    {
        "config": "train/augs/light.json",
        "name": "Light",
    },
    {
        "config": "train/augs/light_corrupt.json",
        "name": "Light + corruption",
    },
    {
        "config": "train/augs/medium.json",
        "name": "Medium",
    },
    {
        "config": "train/augs/medium_corrupt.json",
        "name": "Medium + corruption",
    },
    {
        "config": "train/augs/hard.json",
        "name": "Heavy",
    },
    {
        "config": "train/augs/hard_corrupt.json",
        "name": "Heavy + corruption",
    }
]

_custom_pipeline_path = None
custom_pipeline = None
custom_config = None
gallery1: CompareGallery = None
gallery2: CompareGallery = None
remote_preview_path = "/temp/preview_augs.jpg"

augs_json_config = None
augs_py_preview = None
augs_config_path = os.path.join(g.my_app.data_dir, "augs_config.json")


def _load_template(json_path):
    config = sly.json.load_json_file(json_path)
    pipeline = sly.imgaug_utils.build_pipeline(config["pipeline"], random_order=config["random_order"])  # to validate
    py_code = sly.imgaug_utils.pipeline_to_python(config["pipeline"], config["random_order"])

    return pipeline, py_code, config


def get_aug_templates_list():
    pipelines_info = []
    name_to_py = {}
    for template in _templates:
        json_path = os.path.join(g.root_source_dir, template["config"])
        _, py_code, _ = _load_template(json_path)
        pipelines_info.append({
            **template,
            "py": py_code
        })
        name_to_py[template["name"]] = py_code
    return pipelines_info, name_to_py


def get_template_by_name(name):
    for template in _templates:
        if template["name"] == name:
            json_path = os.path.join(g.root_source_dir, template["config"])
            return _load_template(json_path)
    raise KeyError(f"Template \"{name}\" not found")


def init(data, state):
    state["useAugs"] = True
    state["augsType"] = "template"
    templates_info, name_to_py = get_aug_templates_list()
    data["augTemplates"] = templates_info
    data["augPythonCode"] = name_to_py
    state["augsTemplateName"] = templates_info[2]["name"]
    _, py_code, config = get_template_by_name(state["augsTemplateName"])
    global augs_json_config, augs_py_preview
    augs_json_config = config
    augs_py_preview = py_code

    data["pyViewOptions"] = {
        "mode": 'ace/mode/python',
        "showGutter": False,
        "readOnly": True,
        "maxLines": 100,
        "highlightActiveLine": False
    }

    state["customAugsPath"] = ""
    data["customAugsPy"] = None

    global gallery1, gallery2
    gallery1 = CompareGallery(g.task_id, g.api, "data.gallery1", g.project_meta)
    data["gallery1"] = gallery1.to_json()
    gallery2 = CompareGallery(g.task_id, g.api, "data.gallery2", g.project_meta)
    data["gallery2"] = gallery2.to_json()
    state["collapsed4"] = True
    state["disabled4"] = True
    data["done4"] = False


def restart(data, state):
    data["done4"] = False


@g.my_app.callback("load_existing_pipeline")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def load_existing_pipeline(api: sly.Api, task_id, context, state, app_logger):
    global _custom_pipeline_path, custom_pipeline, custom_config

    api.task.set_field(task_id, "data.customAugsPy", None)

    remote_path = state["customAugsPath"]
    _custom_pipeline_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(remote_path))
    api.file.download(g.team_id, remote_path, _custom_pipeline_path)

    custom_pipeline, py_code, custom_config = _load_template(_custom_pipeline_path)
    api.task.set_field(task_id, "data.customAugsPy", py_code)


@g.my_app.callback("preview_augs")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def preview_augs(api: sly.Api, task_id, context, state, app_logger):
    global gallery1, gallery2
    ds_name, item_name = input_project.get_random_item()
    image_info = input_project.get_image_info_from_cache(ds_name, item_name)

    if state["augsType"] == "template":
        gallery = gallery1
        augs_ppl, _, _ = get_template_by_name(state["augsTemplateName"])
    else:
        gallery = gallery2
        augs_ppl = custom_pipeline

    img = api.image.download_np(image_info.id)
    ann_json = api.annotation.download(image_info.id).annotation
    ann = sly.Annotation.from_json(ann_json, g.project_meta)
    gallery.set_left("before", image_info.path_original, ann)
    _, res_img, res_ann = sly.imgaug_utils.apply(augs_ppl, g.project_meta, img, ann)
    local_image_path = os.path.join(g.my_app.data_dir, "preview_augs.jpg")
    sly.image.write(local_image_path, res_img)
    if api.file.exists(g.team_id, remote_preview_path):
        api.file.remove(g.team_id, remote_preview_path)
    file_info = api.file.upload(g.team_id, local_image_path, remote_preview_path)
    gallery.set_right("after", file_info.storage_path, res_ann)
    gallery.update(options=False)


@g.my_app.callback("use_augs")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def use_augs(api: sly.Api, task_id, context, state, app_logger):
    global augs_config_path
    global augs_json_config
    global augs_py_preview
    global custom_config

    if state["useAugs"]:
        if state["augsType"] == "template":
            _, py_code, config = get_template_by_name(state["augsTemplateName"])
        else:
            if custom_config is None:
                raise Exception("Please, load the augmentations by clicking on the \"LOAD\" button.")
            config = custom_config

        augs_json_config = config
        sly.json.dump_json_file(augs_json_config, augs_config_path)
        sly.json.dump_json_file(augs_json_config, os.path.join(g.info_dir, "augs_config.json"))
        # augs_py_preview = py_code
        # augs_py_path = os.path.join(g.my_app.data_dir, "augs_preview.py")
        # with open(augs_py_path, 'w') as f:
        #     f.write(augs_py_preview)
        
    else:
        augs_config_path = None

    fields = [
        {"field": "data.done4", "payload": True},
        {"field": "state.collapsed5", "payload": False},
        {"field": "state.disabled5", "payload": False},
        {"field": "state.activeStep", "payload": 5},
    ]
    g.api.app.set_fields(g.task_id, fields)

    print(augs_json_config)
    print(augs_config_path)