import json
import os
import pathlib
import sys

import supervisely as sly


root_source_path = str(pathlib.Path(sys.argv[0]).parents[2])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)

models_configs_dir = os.path.join(root_source_path, "model_configs")
print(f"Models configs directory: {models_configs_dir}")
sys.path.append(models_configs_dir)

my_app = sly.AppService()
api = my_app.public_api

TASK_ID = my_app.task_id
TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])

meta: sly.ProjectMeta = None
model = None
local_weights_path = None
model_config_local_path = None
cfg = None
dataset = None
device = None