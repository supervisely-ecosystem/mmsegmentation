import os
import random
from collections import namedtuple
import supervisely as sly
from supervisely.project.download import is_cached
import sly_globals as g
from sly_train_progress import get_progress_cb, reset_progress, init_progress
from sly_project_cached import download_project, validate_project

progress_index = 1
_images_infos = None  # dataset_name -> image_name -> image_info
_cache_base_filename = os.path.join(g.my_app.data_dir, "images_info")
_cache_path = _cache_base_filename + ".db"
project_fs: sly.Project = None
_image_id_to_paths = {}


def init_selector_state(state: dict):
    state.update(
        {
            "selectWorkspace": {"teamId": g.team_id, "workspaceId": g.workspace_id},
            "selectProject": {"value": g.project_id},
            "selectDataset": {
                "options": {
                    "multiple": True,
                    "fit-input-width": True,
                },
                "value": [g.dataset_id] if g.dataset_id else None,
            },
            "selectAllDatasets": g.select_all_datasets,
        }
    )


def init_selector_data(data):
    workspace_info = g.workspace_info
    project_infos = g.api.project.get_list(
        workspace_info.id, filters=[{"field": "type", "operator": "=", "value": "images"}]
    )
    project_options = [
        {"value": info.id, "label": info.name, "disabled": False} for info in project_infos
    ]
    ds_items = g.generate_selector_items_from_tree(g.dataset_tree)
    data.update(
        {
            "selectWorkspace": {
                "hide": False,
                "loading": False,
                "disabled": True,
                "options": [
                    {"id": workspace_info.id, "name": workspace_info.name},
                ],
            },
            "selectProject": {
                "hide": False,
                "loading": False,
                "disabled": False,
                "items": project_options,
            },
            "selectDataset": {
                "disabled": False,
                "items": ds_items,
                "width": 350,
            },
            "selectAllDatasets": {
                "disabled": False,
            },
        }
    )


def init(data, state):
    init_selector_state(state)
    init_selector_data(data)

    init_progress(progress_index, data)
    data["done1"] = False
    state["collapsed1"] = False
    data["isCached"] = is_cached(g.project_info.id)
    state["useCache"] = True


@g.my_app.callback("download_project")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def download(api: sly.Api, task_id, context, state, app_logger):
    fields_to_disable = [
        "data.selectProject",
        "data.selectDataset",
        "data.selectAllDatasets",
    ]
    fields_enabled = api.app.get_fields(task_id, fields_to_disable)
    fields_disabled = fields_enabled.copy()
    for field in fields_to_disable:
        fields_disabled[field]["disabled"] = True

    f = [{"field": field, "payload": fields_disabled[field]} for field in fields_to_disable]
    api.app.set_fields(task_id, f)

    if state["selectAllDatasets"]:
        datasets = g.datasets
    else:
        ds_ids = state["selectDataset"]["value"]
        datasets = g.filter_datasets_aggregated(ds_ids)

    if len(datasets) == 0:
        raise RuntimeError("No datasets selected. Please select at least one dataset.")

    try:
        if sly.fs.dir_exists(g.project_dir):
            pass
        else:
            use_cache = state["useCache"]
            sly.fs.mkdir(g.project_dir)
            download_project(
                api=g.api,
                project_info=g.project_info,
                dataset_infos=datasets,
                project_dir=g.project_dir,
                use_cache=use_cache,
                progress_index=progress_index,
            )
            reset_progress(progress_index)
            if use_cache:
                try:
                    validate_project(g.project_dir)
                except Exception:
                    app_logger.warning(
                        "Cache is corrupted. Downloading project without cache.", exc_info=True
                    )
                    download_project(
                        api=g.api,
                        project_info=g.project_info,
                        project_dir=g.project_dir,
                        dataset_infos=datasets,
                        use_cache=False,
                        progress_index=progress_index,
                    )
                    reset_progress(progress_index)

        global project_fs
        project_fs = sly.Project(g.project_dir, sly.OpenMode.READ)
    except Exception as e:
        reset_progress(progress_index)
        raise e

    filtered_tree = g.filter_tree_by_ids(g.dataset_tree, [ds.id for ds in datasets])
    available_datasets = g.generate_selector_items_from_tree(filtered_tree)
    ds_selector_data = {
        "hide": False,
        "loading": False,
        "disabled": False,
        "width": 350,
        "items": available_datasets,
    }
    fields = [
        {"field": "data.done1", "payload": True},
        {"field": "state.collapsed2", "payload": False},
        {"field": "state.disabled2", "payload": False},
        {"field": "state.activeStep", "payload": 2},
        {"field": "data.trainDatasetSelector", "payload": ds_selector_data},
        {"field": "data.valDatasetSelector", "payload": ds_selector_data},
    ]
    # fields.extend([{"field": field, "payload": fields_enabled[field]} for field in fields_to_disable])
    api.app.set_fields(g.task_id, fields)


@g.my_app.callback("change_project")
def select_project_value_changed(api: sly.Api, task_id, context, state, app_logger):
    try:
        project_value = state.get("selectProject", {}).get("value")
        if project_value is None:
            sly.logger.warning("Could not update project.")
        g.update_project(project_value)
        ds_selector_data = api.app.get_field(task_id, "data.selectDataset")
        ds_selector_data["items"] = g.generate_selector_items_from_tree(g.dataset_tree)
        api.app.set_field(task_id, "data.selectDataset", ds_selector_data)
        init_selector_state(state)
    except Exception as e:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=str(e))


def get_image_info_from_cache(dataset_name, item_name):
    dataset_fs = project_fs.datasets.get(dataset_name)
    img_info_path = dataset_fs.get_img_info_path(item_name)
    image_info_dict = sly.json.load_json_file(img_info_path)
    ImageInfo = namedtuple("ImageInfo", image_info_dict)
    info = ImageInfo(**image_info_dict)

    # add additional info - helps to save split paths to txt files
    _image_id_to_paths[info.id] = dataset_fs.get_item_paths(item_name)._asdict()

    return info


def get_paths_by_image_id(image_id):
    return _image_id_to_paths[image_id]


def get_random_item():
    global project_fs
    project_fs = sly.Project(g.project_dir, sly.OpenMode.READ)

    all_ds_names = project_fs.datasets.keys()
    non_empty_ds = [ds for ds in all_ds_names if len(project_fs.datasets.get(ds)) > 0]
    if len(non_empty_ds) == 0:
        raise ValueError("No images in the project")
    ds_name = random.choice(non_empty_ds)
    ds = project_fs.datasets.get(ds_name)
    items = list(ds)
    item_name = random.choice(items)
    return ds_name, item_name
