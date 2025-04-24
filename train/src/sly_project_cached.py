import os

import supervisely as sly
from supervisely.project.download import (
    download_to_cache,
    copy_from_cache,
    get_cache_size,
    download_async
)
from sly_train_progress import get_progress_cb
import sly_globals as g


def _no_cache_download(api: sly.Api, project_info: sly.ProjectInfo, dataset_ids: list, project_dir: str, progress_index: int, total: int = None):
    try:
        download_progress = get_progress_cb(progress_index, "Downloading input data...", total)
        download_async(
            api,
            project_info.id,
            project_dir,
            dataset_ids=dataset_ids,
            progress_cb=download_progress
        )
    except Exception as e:
        api.logger.warning(
            "Failed to download project using async download. Trying sync download..."
        )
        download_progress = get_progress_cb(progress_index, "Downloading input data...", total)
        sly.download(
            api=api,
            project_id=project_info.id,
            dest_dir=project_dir,
            dataset_ids=dataset_ids,
            log_progress=True,
            progress_cb=download_progress,
        )

def download_project(
    api: sly.Api,
    project_info: sly.ProjectInfo,
    dataset_infos,
    project_dir: str,
    use_cache: bool,
    progress_index: int,
):
    total = project_info.items_count if dataset_infos is None else sum((ds.items_count for ds in dataset_infos))
    ds_ids = [ds.id for ds in dataset_infos] if dataset_infos else None

    if os.path.exists(project_dir):
        sly.fs.clean_dir(project_dir)
    if not use_cache:
        _no_cache_download(api, project_info, ds_ids, project_dir, progress_index, total)
        return

    try:
        # download
        download_progress = get_progress_cb(progress_index, "Downloading input data...", total)
        download_to_cache(
            api=api,
            project_id=project_info.id,
            dataset_infos=dataset_infos,
            log_progress=True,
            progress_cb=download_progress,
        )
        # copy datasets from cache
        total = get_cache_size(project_info.id)
        download_progress = get_progress_cb(progress_index, "Retreiving data from cache...", total, is_size=True)
        # ds_names = [ds.name for ds in dataset_infos] if dataset_infos else None
        ds_paths = [g.id_to_aggregated_name[ds.id] for ds in dataset_infos] if dataset_infos else None
        copy_from_cache(
            project_id=project_info.id,
            dest_dir=project_dir,
            dataset_paths=ds_paths,
            progress_cb=download_progress,
        )
    except Exception as e:
        sly.logger.debug(e)
        sly.logger.warning(f"Failed to retreive project from cache. Downloading it...", exc_info=True)
        if os.path.exists(project_dir):
            sly.fs.clean_dir(project_dir)
        _no_cache_download(api, project_info, ds_ids, project_dir, progress_index, total)


def validate_project(project_dir):
    """Iterate over project and try to open annotations"""
    project_fs = sly.Project(project_dir, sly.OpenMode.READ)
    for dataset in project_fs:
        dataset: sly.Dataset
        for item_name in dataset:
            dataset.get_item_path(item_name)
            dataset.get_ann(item_name, project_fs.meta)
