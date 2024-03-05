import os

import supervisely as sly
from supervisely.project.download import (
    download_to_cache,
    copy_from_cache,
    is_cached,
    get_cache_size,
)
from sly_train_progress import get_progress_cb
import sly_globals as g


def download_project(
    api: sly.Api,
    project_info: sly.ProjectInfo,
    project_dir: str,
    use_cache: bool,
    progress_index: int,
):
    if os.path.exists(project_dir):
        sly.fs.clean_dir(project_dir)
    if not use_cache:
        total = project_info.items_count
        download_progress = get_progress_cb(progress_index, "Downloading input data...", total * 2)
        sly.download(
            api=api,
            project_id=project_info.id,
            dest_dir=project_dir,
            dataset_ids=None,
            log_progress=True,
            progress_cb=download_progress,
        )
        return

    # get datasets to download and cached
    dataset_infos = api.dataset.get_list(project_info.id)
    to_download = [info for info in dataset_infos if not is_cached(project_info.id, info.name)]
    cached = [info for info in dataset_infos if is_cached(project_info.id, info.name)]
    if len(cached) == 0:
        log_msg = "No cached datasets found"
    else:
        log_msg = "Using cached datasets: " + ", ".join(
            f"{ds_info.name} ({ds_info.id})" for ds_info in cached
        )
    sly.logger.info(log_msg)
    if len(to_download) == 0:
        log_msg = "All datasets are cached. No datasets to download"
    else:
        log_msg = "Downloading datasets: " + ", ".join(
            f"{ds_info.name} ({ds_info.id})" for ds_info in to_download
        )
    sly.logger.info(log_msg)
    # get images count
    total = sum([ds_info.images_count for ds_info in to_download])
    # download
    if total > 0:
        download_progress = get_progress_cb(progress_index, "Downloading input data...", total * 2)
        download_to_cache(
            api=api,
            project_id=project_info.id,
            dataset_infos=to_download,
            log_progress=True,
            progress_cb=download_progress,
        )
    # copy datasets from cache
    total = sum([get_cache_size(project_info.id, ds.name) for ds in dataset_infos])
    dataset_names = [ds_info.name for ds_info in dataset_infos]
    if total > 0:
        download_progress = get_progress_cb(progress_index, "Retreiving data from cache...", total, is_size=True)
        copy_from_cache(
            project_id=project_info.id,
            dest_dir=project_dir,
            dataset_names=dataset_names,
            progress_cb=download_progress,
        )
