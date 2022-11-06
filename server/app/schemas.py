from enum import Enum
from typing import Optional

from pydantic import BaseModel


class Operations(str, Enum):
    classi = "classification"
    detect = "detection"


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
    result: dict
    time: float
    net: str
    battery: Optional[float]


class PredictBody(BaseModel):
    image: bytes
    time: float
    battery: Optional[float]


class PredictResponse(BaseModel):
    image: bytes
    time: float
