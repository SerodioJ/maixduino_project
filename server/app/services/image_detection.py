import torch

from schemas import DetectionModel


class ImageDetection:
    def __init__(self, model: DetectionModel) -> None:
        self.model = torch.hub.load("ultralytics/yolov5", model, pretrained=True)

    def predict(self, img):
        preds = self.model([img])
        preds.pandas().xyxy[0].to_json(orient="records")

        return preds.pandas().xyxy[0].to_json(orient="records")
