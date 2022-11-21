import torch


class ImageDetection:
    def __init__(self, model: str) -> None:
        self.model = torch.hub.load("ultralytics/yolov5", model, pretrained=True)


    def predict(self, img):
        preds = self.model([img])
        df = preds.pandas().xyxy[0]
        results = df.loc[df['class']==0].to_numpy()
        r_list = []
        for result in results:
            r_list.append(
                {
                    "x": round(result[0]),
                    "y": round(result[1]),
                    "w": round(result[2]-result[0]),
                    "h": round(result[3]-result[1]),
                    "value": result[4],
                    "classid": result[5],
                    "name": result[6]
                }
            ) 

        return r_list
