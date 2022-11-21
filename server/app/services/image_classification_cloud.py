import importlib
import numpy as np

from schemas import ClassificationModel
from settings import settings


class ImageClassification:
    def __init__(self, model: ClassificationModel) -> None:
        self.model = settings.class_model_app[model]()
        self.model_lib = importlib.import_module(settings.class_model_lib[model])

    def predict(self, img, top: int = 1):
        model_input = np.asarray(img)
        model_input = np.expand_dims(model_input, axis=0)
        model_input = self.model_lib.preprocess_input(model_input)

        preds = self.model.predict(model_input)

        results = self.model_lib.decode_predictions(preds, top=top)[0]
        data = {}
        for value in results:
            data[value[0]] = [value[1], float(value[2])]
        return data
