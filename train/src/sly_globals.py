import os
from pathlib import Path
import sys
import supervisely as sly
from supervisely.app.v1.app_service import AppService
import shutil
import pkg_resources

my_app = AppService()
api = my_app.public_api
task_id = my_app.task_id

team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])
project_id = int(os.environ['modal.state.slyProjectId'])

project_info = api.project.get_info_by_id(project_id)
if project_info is None:  # for debug
    raise ValueError(f"Project with id={project_id} not found")

sly.fs.clean_dir(my_app.data_dir)  # @TODO: for debug

project_dir = os.path.join(my_app.data_dir, "sly_project")
project_seg_dir = os.path.join(my_app.data_dir, "sly_seg_project")
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

artifacts_dir = os.path.join(my_app.data_dir, "artifacts")
sly.fs.mkdir(artifacts_dir)
info_dir = os.path.join(artifacts_dir, "info")
sly.fs.mkdir(info_dir)
checkpoints_dir = os.path.join(artifacts_dir, "checkpoints")
sly.fs.mkdir(checkpoints_dir)

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

configs_dir = os.path.join(root_source_dir, "configs")
mmseg_ver = pkg_resources.get_distribution("mmsegmentation").version
if os.path.isdir(f"/tmp/mmseg/mmsegmentation-{mmseg_ver}"):
    if os.path.isdir(configs_dir):
        shutil.rmtree(configs_dir)
    sly.logger.info(f"Getting model configs of current mmsegmentation version {mmseg_ver}...")
    shutil.copytree(f"/tmp/mmseg/mmsegmentation-{mmseg_ver}/configs", configs_dir)
    models_cnt = len(os.listdir(configs_dir)) - 1
    sly.logger.info(f"Found {models_cnt} models in {configs_dir} directory.")
