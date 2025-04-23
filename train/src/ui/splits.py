import supervisely as sly
import sly_globals as g
import os

train_set = None
val_set = None

train_set_path = os.path.join(g.my_app.data_dir, "train.txt")
val_set_path = os.path.join(g.my_app.data_dir, "val.txt")


def init(project_info, project_meta: sly.ProjectMeta, data, state):
    data["selectTrainDataset"] = {"hide": False, "loading": False}
    data["selectContainer"] = {"hide": False, "loading": False}

    data["trainDatasetSelector"] = {}
    data["valDatasetSelector"] = {}
    state["trainDatasetSelector"] = {"options": {"multiple": True}}
    state["valDatasetSelector"] = {"options": {"multiple": True}}

    data["randomSplit"] = [
        {"name": "train", "type": "success"},
        {"name": "val", "type": "primary"},
        {"name": "total", "type": "gray"},
    ]
    data["totalImagesCount"] = project_info.items_count

    train_percent = 80
    train_count = int(project_info.items_count / 100 * train_percent)
    state["randomSplit"] = {
        "count": {
            "total": project_info.items_count,
            "train": train_count,
            "val": project_info.items_count - train_count,
        },
        "percent": {"total": 100, "train": train_percent, "val": 100 - train_percent},
        "shareImagesBetweenSplits": False,
        "sliderDisabled": False,
    }

    state["splitMethod"] = "random"

    state["trainTagName"] = ""
    if project_meta.tag_metas.get("train") is not None:
        state["trainTagName"] = "train"
    state["valTagName"] = ""
    if project_meta.tag_metas.get("val") is not None:
        state["valTagName"] = "val"

    state["trainDatasets"] = []
    state["valDatasets"] = []
    state["untaggedImages"] = "train"
    state["splitInProgress"] = False
    state["trainImagesCount"] = None
    state["valImagesCount"] = None
    data["done2"] = False
    state["collapsed2"] = True
    state["disabled2"] = True


def get_train_val_sets(project_dir, state):
    split_method = state["splitMethod"]
    if split_method == "random":
        train_count = state["randomSplit"]["count"]["train"]
        val_count = state["randomSplit"]["count"]["val"]
        train_set, val_set = sly.Project.get_train_val_splits_by_count(
            project_dir, train_count, val_count
        )
        return train_set, val_set
    elif split_method == "tags":
        train_tag_name = state["trainTagName"]
        val_tag_name = state["valTagName"]
        add_untagged_to = state["untaggedImages"]
        train_set, val_set = sly.Project.get_train_val_splits_by_tag(
            project_dir, train_tag_name, val_tag_name, add_untagged_to
        )
        return train_set, val_set
    elif split_method == "datasets":
        train_names, val_names = [], []
        for ds_id in state["trainDatasetSelector"]["value"]:
            ids = [ds_id] + [children.id for children in g.parent_to_children.get(ds_id, [])]
            for i in ids:
                train_names.append(g.id_to_aggregated_name[i])

        for ds_id in state["valDatasetSelector"]["value"]:
            ids = [ds_id] + [children.id for children in g.parent_to_children.get(ds_id, [])]
            for i in ids:
                val_names.append(g.id_to_aggregated_name[i])
        train_set, val_set = sly.Project.get_train_val_splits_by_dataset(
            project_dir, train_names, val_names
        )
        return train_set, val_set
    else:
        raise ValueError(f"Unknown split method: {split_method}")


def verify_train_val_sets(train_set, val_set):
    if len(train_set) == 0:
        g.my_app.show_modal_window(
            "Train set is empty, check or change split configuration", level="warning"
        )
        return False
    if len(val_set) == 0:
        g.my_app.show_modal_window(
            "Val set is empty, check or change split configuration", level="warning"
        )
        return False
    return True


def set_dataset_ind_to_items(project_dir):
    global project_fs
    project_fs = sly.Project(project_dir, sly.OpenMode.READ)
    ds_cnt = 0
    for dataset in project_fs.datasets:
        dataset: sly.Dataset
        for name in dataset.get_items_names():
            new_name = f"{ds_cnt}_{name}"
            img_path, ann_path = dataset.get_item_paths(name)
            img_info_path = dataset.get_item_info_path(name)
            if sly.fs.file_exists(img_path):
                os.rename(img_path, img_path.replace(name, new_name))
            if sly.fs.file_exists(ann_path):
                ann_name = sly.fs.get_file_name(ann_path)
                new_ann_name = f"{ds_cnt}_{ann_name}"
                os.rename(ann_path, ann_path.replace(ann_name, new_ann_name))
            if sly.fs.file_exists(img_info_path):
                img_info_name = sly.fs.get_file_name(img_info_path)
                new_img_info_name = f"{ds_cnt}_{img_info_name}"
                os.rename(img_info_path, img_info_path.replace(img_info_name, new_img_info_name))
        ds_cnt += 1
    project_fs = sly.Project(project_dir, sly.OpenMode.READ)


@g.my_app.callback("create_splits")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def create_splits(api: sly.Api, task_id, context, state, app_logger):
    step_done = False
    global train_set, val_set
    try:
        api.task.set_field(task_id, "state.splitInProgress", True)
        # to support duplicates
        set_dataset_ind_to_items(g.project_dir)
        train_set, val_set = get_train_val_sets(g.project_dir, state)
        sly.logger.info(f"Train set: {len(train_set)} images")
        sly.logger.info(f"Val set: {len(val_set)} images")
        success = verify_train_val_sets(train_set, val_set)
        if not success:
            api.task.set_field(task_id, "state.splitInProgress", False)
            return
        step_done = True
    except Exception as e:
        train_set = None
        val_set = None
        step_done = False
        raise e
    finally:
        api.task.set_field(task_id, "state.splitInProgress", False)
        fields = [
            {"field": "state.splitInProgress", "payload": False},
            {"field": f"data.done2", "payload": step_done},
            {
                "field": f"state.trainImagesCount",
                "payload": None if train_set is None else len(train_set),
            },
            {
                "field": f"state.valImagesCount",
                "payload": None if val_set is None else len(val_set),
            },
        ]
        if step_done is True:
            fields.extend(
                [
                    {"field": "state.collapsed3", "payload": False},
                    {"field": "state.disabled3", "payload": False},
                    {"field": "state.activeStep", "payload": 3},
                ]
            )
        g.api.app.set_fields(g.task_id, fields)
    if train_set is not None:
        _save_set(train_set_path, train_set)
    if val_set is not None:
        _save_set(val_set_path, val_set)


def _save_set(save_path, items):
    with open(os.path.join(save_path), "w") as f:
        f.writelines(item.name + "\n" for item in items)
