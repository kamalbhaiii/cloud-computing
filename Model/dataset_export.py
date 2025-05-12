# pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="uUiAY4ufKNxPGS9jwXP6")
project = rf.workspace("yolo-wea97").project("new_datasets_cloud_computing")
version = project.version(2)
dataset = version.download("yolov8")