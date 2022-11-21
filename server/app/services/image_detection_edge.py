# Adapted from https://github.com/ultralytics/yolov5/blob/master/models/common.py

import contextlib
import zipfile
import ast
import os

import numpy as np

from .utils import non_max_suppression, letterbox

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter


class ImageDetection:
    def __init__(self, model: str) -> None:
        path = os.path.join("models", f"{model}-fp16.tflite")
        interpreter = Interpreter(model_path=path)
        interpreter.allocate_tensors()  # allocate
        input_details = interpreter.get_input_details()  # inputs
        output_details = interpreter.get_output_details()  # outputs
        stride = 32
        names = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            8: "boat",
            9: "traffic light",
            10: "fire hydrant",
            11: "stop sign",
            12: "parking meter",
            13: "bench",
            14: "bird",
            15: "cat",
            16: "dog",
            17: "horse",
            18: "sheep",
            19: "cow",
            20: "elephant",
            21: "bear",
            22: "zebra",
            23: "giraffe",
            24: "backpack",
            25: "umbrella",
            26: "handbag",
            27: "tie",
            28: "suitcase",
            29: "frisbee",
            30: "skis",
            31: "snowboard",
            32: "sports ball",
            33: "kite",
            34: "baseball bat",
            35: "baseball glove",
            36: "skateboard",
            37: "surfboard",
            38: "tennis racket",
            39: "bottle",
            40: "wine glass",
            41: "cup",
            42: "fork",
            43: "knife",
            44: "spoon",
            45: "bowl",
            46: "banana",
            47: "apple",
            48: "sandwich",
            49: "orange",
            50: "broccoli",
            51: "carrot",
            52: "hot dog",
            53: "pizza",
            54: "donut",
            55: "cake",
            56: "chair",
            57: "couch",
            58: "potted plant",
            59: "bed",
            60: "dining table",
            61: "toilet",
            62: "tv",
            63: "laptop",
            64: "mouse",
            65: "remote",
            66: "keyboard",
            67: "cell phone",
            68: "microwave",
            69: "oven",
            70: "toaster",
            71: "sink",
            72: "refrigerator",
            73: "book",
            74: "clock",
            75: "vase",
            76: "scissors",
            77: "teddy bear",
            78: "hair drier",
            79: "toothbrush",
        }
        self.__dict__.update(locals())

    def forward(self, img):
        b, h, w, ch = img.shape  # batch, channel, height, width
        input = self.input_details[0]
        self.interpreter.set_tensor(input["index"], img)
        self.interpreter.invoke()
        y = []
        for output in self.output_details:
            x = self.interpreter.get_tensor(output["index"])
            y.append(x)
        y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
        y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels
        return y

    def preprocess(self, img):

        im = letterbox(np.array(img).astype(np.float32))[0]
        im = np.ascontiguousarray(im)  # contiguous
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        return im

    def predict(self, img):
        resized_img = self.preprocess(img)
        predictions = self.forward(resized_img)
        results = non_max_suppression(predictions)
        formated_results = []
        for elem in results[0]:
            formated_results.append(
                {
                    "x": int(elem[0]),
                    "y": int(elem[1]),
                    "w": int(elem[2]) - int(elem[0]),
                    "h": int(elem[3]) - int(elem[1]),
                    "value": float(elem[4]),
                    "classid": int(elem[5]),
                    "name": self.names[int(elem[5])],
                }
            )


        return formated_results
