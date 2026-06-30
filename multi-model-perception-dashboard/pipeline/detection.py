"""YOLOv8 object detection: batched preprocess / inference / postprocess + draw.

Adapted from the source repo `ONNX-YOLOv8-Object-Detection/yolov8/` and made
batch-aware so the two camera frames can flow through as one logical batch.

Pipeline IR mapping:
    node 4  preprocess  -> Detector.preprocess(frames)   -> (N,3,H,W) fp32
    node 5  inference   -> Detector.infer(tensor)        -> raw outputs (per frame)
    node 6  postprocess -> Detector.postprocess(raw, frames) -> list of (boxes, scores, ids)

Note on batching: the bundled `yolov8m.onnx` has a *fixed* batch dim of 1, so
`infer()` runs the session once per frame. If you drop in a dynamic-batch
export, it issues a single batched `session.run` instead.
"""

import time

import numpy as np
import cv2
import onnxruntime as ort

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))


# ---------- NMS helpers (from repo utils.py) ----------
def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def compute_iou(box, boxes):
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return intersection / (box_area + boxes_area - intersection)


def nms(boxes, scores, iou_threshold):
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ious = compute_iou(boxes[i, :], boxes[order[1:], :])
        order = order[np.where(ious < iou_threshold)[0] + 1]
    return keep


def multiclass_nms(boxes, scores, class_ids, iou_threshold):
    keep = []
    for cid in np.unique(class_ids):
        idx = np.where(class_ids == cid)[0]
        k = nms(boxes[idx, :], scores[idx], iou_threshold)
        keep.extend(idx[k])
    return keep


# ---------- Batched detector ----------
class Detector:
    def __init__(self, onnx_path, conf_thres=0.5, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        avail = ort.get_available_providers()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        shape = self.session.get_inputs()[0].shape  # [N, 3, H, W]
        self.in_h = shape[2] if isinstance(shape[2], int) else 640
        self.in_w = shape[3] if isinstance(shape[3], int) else 640
        # Static batch dim of 1 => loop; otherwise issue one batched run.
        self.static_batch1 = (shape[0] == 1)
        self.infer_ms = 0.0

    # IR node 4: preprocess  (BGR list -> (N,3,H,W) fp32 RGB)
    def preprocess(self, frames):
        tensors = []
        for bgr in frames:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self.in_w, self.in_h)) / 255.0
            tensors.append(rgb.transpose(2, 0, 1))
        return np.asarray(tensors, dtype=np.float32)

    # IR node 5: inference  (returns list of per-frame raw outputs)
    def infer(self, batch_tensor):
        t0 = time.perf_counter()
        if self.static_batch1:
            raws = [self.session.run(self.output_names, {self.input_name: batch_tensor[i:i + 1]})[0]
                    for i in range(batch_tensor.shape[0])]
        else:
            out = self.session.run(self.output_names, {self.input_name: batch_tensor})[0]
            raws = [out[i:i + 1] for i in range(out.shape[0])]
        self.infer_ms = (time.perf_counter() - t0) * 1000.0
        return raws

    # IR node 6: postprocess (decode + threshold + rescale + NMS), per frame
    def postprocess(self, raws, frames):
        dets = []
        for raw, frame in zip(raws, frames):
            dets.append(self._decode_one(raw, frame.shape[1], frame.shape[0]))
        return dets

    def _decode_one(self, output, img_w, img_h):
        predictions = np.squeeze(output[0]).T          # (8400, 84)
        scores = np.max(predictions[:, 4:], axis=1)
        keep = scores > self.conf_threshold
        predictions, scores = predictions[keep], scores[keep]
        if len(scores) == 0:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = predictions[:, :4]
        # rescale from model input space to original image
        boxes = boxes / np.array([self.in_w, self.in_h, self.in_w, self.in_h], dtype=np.float32)
        boxes *= np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
        boxes = xywh2xyxy(boxes)
        idx = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)
        return boxes[idx], scores[idx], class_ids[idx]


# ---------- Drawing (from repo utils.py) ----------
def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    det_img = image.copy()
    img_h, img_w = image.shape[:2]
    font_size = min(img_h, img_w) * 0.0006
    text_thickness = int(min(img_h, img_w) * 0.001)

    # translucent fills
    mask_img = det_img.copy()
    for box, cid in zip(boxes, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), colors[cid], -1)
    det_img = cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)

    for cid, box, score in zip(class_ids, boxes, scores):
        color = colors[cid]
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)
        caption = f'{class_names[cid]} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_size, text_thickness)
        th = int(th * 1.2)
        cv2.rectangle(det_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
        cv2.putText(det_img, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, (255, 255, 255), text_thickness, cv2.LINE_AA)
    return det_img
