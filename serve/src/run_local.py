from supervisely.nn.inference import Session
from supervisely import Api
from supervisely.imaging.image import read
import os
from pathlib import Path
from dotenv import load_dotenv

root_source_path = str(Path(__file__).parents[2])
app_source_path = str(Path(__file__).parents[1])
load_dotenv(os.path.join(app_source_path, "local.env"))
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = Api()
session = Session(api, session_url="http://0.0.0.0:8000")

image_path = "./demo_data/image_01.jpg"
result_ann = session.inference_image_path(image_path=image_path)
vis_path = "./demo_data/image_01_prediction.jpg"
image_np = read(image_path)
result_ann.draw_pretty(
    bitmap=image_np, thickness=0, output_path=vis_path, fill_rectangles=False
)
print(f"predictions and visualization have been saved: {vis_path}")