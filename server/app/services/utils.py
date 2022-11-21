# Adapted from https://github.com/ultralytics/yolov5

import numpy as np
import time
import cv2


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def IOU(bboxes1, bboxes2):
    # adapted from https://towardsdatascience.com/intuition-and-implementation-of-non-max-suppression-algorithm-in-object-detection-d68ba938b630
    bboxes1 = [int(i) for i in bboxes1]
    bboxes2= [int(i) for i in bboxes2]

    xA = max(bboxes1[0], bboxes2[0])
    yA = max(bboxes1[1], bboxes2[1])
    xB = min(bboxes1[2], bboxes2[2])
    yB = min(bboxes1[3], bboxes2[3])

    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    box1_area = (bboxes1[2] - bboxes1[0] + 1) * (bboxes1[3] - bboxes1[1] + 1)
    box2_area = (bboxes2[2] - bboxes2[0] + 1) * (bboxes2[3] - bboxes2[1] + 1)

    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou

def nms(bboxes_array, scores, classes, threshold):
# adapted from https://towardsdatascience.com/intuition-and-implementation-of-non-max-suppression-algorithm-in-object-detection-d68ba938b630
    final_boxes = np.ones(bboxes_array.shape[0])
    for curr in range(bboxes_array.shape[0]):
        # removing the best probability bounding box
        if not final_boxes[curr]:
            continue
        box = bboxes_array[curr]
        for comp in range(curr+1, bboxes_array.shape[0]):
            if  final_boxes[comp] and classes[curr] == classes[comp]:
                iou = IOU(bboxes_array[curr], bboxes_array[comp])
                if iou >= threshold:
                    final_boxes[comp] = 0
    return final_boxes == 1





def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    if isinstance(
        prediction, (list, tuple)
    ):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [np.zeros((0, 6 + nm))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = np.zeros((len(lb), nc + nm + 5))
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = np.concatenate((x, v), axis=0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(
            x[:, :4]
        )  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        # if multi_label:
        #     i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
        #     x = np.concatenate(
        #         (box[i], x[i, 5 + j, None], j[:, None], mask[i]), axis=1
        #     )
        # else:  # best class only
        j = np.argmax(x[:, 5:mi], axis=1, keepdims=True)
        conf = np.amax(x[:, 5:mi], axis=1, keepdims=True)
        x = np.concatenate((box, conf, j, mask), 1)[
            conf.flatten() > conf_thres
        ]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[(-x[:, 4]).argsort()[:max_nms]]  # sort by confidence
        else:
            x = x[(-x[:, 4]).argsort()]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4], x[:, 4]  # boxes (offset by class), scores
        i = nms(boxes, scores, c, iou_thres)  # NMS
        

        output[xi] = x[i][:max_det]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded
    return output


def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=False,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)
