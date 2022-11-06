from pydantic import BaseSettings
import tensorflow as tf

from schemas import ClassificationModel, DetectionModel


class Settings(BaseSettings):
    class_model: ClassificationModel = "mobilenet_v2"
    class_model_app = {
        "mobilenet": tf.keras.applications.MobileNet,
        "mobilenet_v2": tf.keras.applications.MobileNetV2,
        "mobilenet_v3small": tf.keras.applications.MobileNetV3Small,
        "mobilenet_v3large": tf.keras.applications.MobileNetV3Large,
        "resnet50": tf.keras.applications.ResNet50,
    }
    class_model_lib = {
        "mobilenet": "tensorflow.keras.applications.mobilenet",
        "mobilenet_v2": "tensorflow.keras.applications.mobilenet_v2",
        "mobilenet_v3small": "tensorflow.keras.applications.mobilenet_v3",
        "mobilenet_v3large": "tensorflow.keras.applications.mobilenet_v3",
        "resnet50": "tensorflow.keras.applications.resnet50",
    }

    detect_model: DetectionModel = "yolov5s"

    model_dict: dict = {"classification": class_model, "detection": detect_model}

    class Config:
        env_file = ".env"


settings = Settings()
