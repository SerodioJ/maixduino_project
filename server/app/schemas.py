from enum import Enum
from typing import Optional

from pydantic import BaseModel


class Operations(str, Enum):
    # classi = "classification"
    detect = "detection"

class Edge(str, Enum):
    cloud = "cloud"
    edge = "edge"
    edge_arm = "edge_arm"


class ClassificationModel(str, Enum):
    mbnet = "mobilenet"
    mbnetv2 = "mobilenet_v2"
    mbnetv3small = "mobilenet_v3small"
    mbnetv3large = "mobilenet_v3large"
    resnet50 = "resnet50"


class DetectionModel(str, Enum):
    v5l = "yolov5l"
    v5m = "yolov5m"
    v5n = "yolov5n"
    v5s = "yolov5s"
    v5x = "yolov5x"


class ResultBody(BaseModel):
    result: list
    time: float
    net: str
    tag: Optional[str]


class PredictBody(BaseModel):
    image: bytes
    time: Optional[float]
    tag: Optional[str]


class PredictResponse(BaseModel):
    image: bytes
    time: float
