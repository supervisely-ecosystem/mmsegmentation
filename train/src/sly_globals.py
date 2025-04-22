import os
from pathlib import Path
import sys
import supervisely as sly
from supervisely.app.v1.app_service import AppService
from supervisely.nn.artifacts.mmsegmentation import MMSegmentation
from typing import List

import shutil
import pkg_resources

def filter_tree_by_ids(tree, ids: List[int]):
    """
    Filters a tree structure by a list of IDs.
    Returns a new tree structure containing only the nodes with IDs in the provided list.

    Args:
        tree (list): A list of dictionaries representing the tree structure.
        ids (list): A list of IDs to filter the tree by.

    Returns:
        list: A filtered tree structure containing only the nodes with IDs in the provided list.
    """
    result = {}
    for ds_info, children in tree.items():
        if ds_info.id in ids:
            result[ds_info] = children
            if children:
                result[ds_info] = filter_tree_by_ids(children, ids)
    return result

def generate_selector_items_from_tree(tree):
    """
    Converts a tree structure into a list of dictionaries suitable for a selector component.
    Each dictionary contains an 'id', 'label', and 'children' keys.

    Args:
        tree (list): A list of dictionaries representing the tree structure.

    Returns:
        list: A list of dictionaries formatted for the selector component.
    """
    result = []
    for ds_info, children in tree.items():
        selector_item = {
            "id": ds_info.id,
            "label": ds_info.name,
        }
        if children:
            selector_item["children"] = generate_selector_items_from_tree(children)
        result.append(selector_item)
    return result

def generate_selector_items_from_list(datasets):
    """
    Converts a list of dataset information into a list of dictionaries suitable for a selector component.
    Each dictionary contains an 'id' and 'label' keys.

    Args:
        datasets (list): A list of dataset information.

    Returns:
        list: A list of dictionaries formatted for the selector component.
    """
    return [{"id": ds.id, "label": ds.name} for ds in datasets]

# from dotenv import load_dotenv

root_source_dir = str(Path(sys.argv[0]).parents[2])
sly.logger.info(f"Root source directory: {root_source_dir}")
sys.path.append(root_source_dir)
source_path = str(Path(sys.argv[0]).parents[0])
sly.logger.info(f"App source directory: {source_path}")
sys.path.append(source_path)
ui_sources_dir = os.path.join(source_path, "ui")
sly.logger.info(f"UI source directory: {ui_sources_dir}")
sys.path.append(ui_sources_dir)
sly.logger.info(f"Added to sys.path: {ui_sources_dir}")

debug_env_path = os.path.join(root_source_dir, "train", "debug.env")
secret_debug_env_path = os.path.join(root_source_dir, "train", "secret_debug.env")
# @TODO: for debug
# load_dotenv(debug_env_path)
# load_dotenv(secret_debug_env_path, override=True)

if sly.is_development():
    from dotenv import load_dotenv
    load_dotenv(debug_env_path)
    load_dotenv(os.path.expanduser("~/supervisely.env"))

my_app = AppService()
api = my_app.public_api
task_id = my_app.task_id

team_id = sly.env.team_id()
team_info = api.team.get_info_by_id(team_id)
workspace_id = sly.env.workspace_id()
workspace_info = api.workspace.get_info_by_id(workspace_id)
project_id = sly.env.project_id(raise_not_found=False)
dataset_id = sly.env.dataset_id(raise_not_found=False)
sly.logger.debug("DATASET_ID: %s, PROJECT_ID: %s", dataset_id, project_id)

try:
    assert bool(project_id) ^ bool(dataset_id)
except AssertionError as e:
    error_msg = "Either project_id or dataset_id should be set, but not both"
    sly.logger.error(error_msg)
    raise RuntimeError(error_msg) from e

if project_id is None:
    project_id = api.dataset.get_info_by_id(dataset_id).project_id
    sly.logger.debug(
        f"Project ID was not set. Using dataset ID {dataset_id} to get project ID {project_id}"
    )

project_info = None
project_meta = None
dataset_tree = None
datasets = []

ds_id_to_info = {}
parent_to_children = {}
id_to_aggregated_name = {}

def get_aggregated_name(ds, ds_id_to_info):
    names = []
    current = ds
    while current:
        names.append(current.name)
        if current.parent_id and current.parent_id in ds_id_to_info:
            current = ds_id_to_info[current.parent_id]
        else:
            break
    return "/".join(reversed(names))


def filter_datasets_aggregated(dataset_ids):
    datasets = []
    for dataset_id in dataset_ids:
        datasets.extend([ds_id_to_info.get(dataset_id)] + parent_to_children.get(dataset_id, []))
    if len(datasets) > len(dataset_ids):
        _ds_names = [ds.name for ds in datasets]
        sly.logger.debug("Aggregated datasets: %s", _ds_names)
    return datasets

def init_project(project_id, dataset_ids=[]):
    global project_info, project_meta, dataset_tree, datasets, ds_id_to_info, parent_to_children, id_to_aggregated_name
    project_info = api.project.get_info_by_id(project_id)
    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    dataset_tree = api.dataset.get_tree(project_id)
    datasets = api.dataset.get_list(project_id, recursive=True)
    ds_id_to_info = {ds.id: ds for ds in datasets}
    id_to_aggregated_name = {ds.id: get_aggregated_name(ds, ds_id_to_info) for ds in ds_id_to_info.values()}
    parent_to_children = {ds.id: [] for ds in datasets}
    for ds in datasets:
        current = ds
        while parent_id := current.parent_id:
            parent_to_children[parent_id].append(ds)
            current = ds_id_to_info[parent_id]

    if dataset_ids:
        datasets = filter_datasets_aggregated(dataset_ids)
        dataset_tree = filter_tree_by_ids(dataset_tree, dataset_ids)

dataset_ids = [dataset_id] if dataset_id else []
init_project(project_id, dataset_ids)
# sly.fs.clean_dir(my_app.data_dir)  # @TODO: for debug

project_dir = os.path.join(my_app.data_dir, "sly_project")
project_seg_dir = os.path.join(my_app.data_dir, "sly_seg_project")
data_dir = sly.app.get_synced_data_dir()
artifacts_dir = os.path.join(data_dir, "artifacts")
sly.fs.mkdir(artifacts_dir)
info_dir = os.path.join(artifacts_dir, "info")
sly.fs.mkdir(info_dir)
checkpoints_dir = os.path.join(artifacts_dir, "checkpoints")
sly.fs.mkdir(checkpoints_dir)

sly_mmseg = MMSegmentation(team_id)

configs_dir = os.path.join(root_source_dir, "configs")
mmseg_ver = pkg_resources.get_distribution("mmsegmentation").version
if os.path.isdir(f"/tmp/mmseg/mmsegmentation-{mmseg_ver}"):
    if os.path.isdir(configs_dir):
        shutil.rmtree(configs_dir)
    sly.logger.info(f"Getting model configs of current mmsegmentation version {mmseg_ver}...")
    shutil.copytree(f"/tmp/mmseg/mmsegmentation-{mmseg_ver}/configs", configs_dir)
    models_cnt = len(os.listdir(configs_dir)) - 1
    sly.logger.info(f"Found {models_cnt} folders in {configs_dir} directory.")

sly_mmseg_generated_metadata = None # for project Workflow purposes

def update_project(project_id: int):
    init_project(project_id)    
    sly.logger.info("Project updated")