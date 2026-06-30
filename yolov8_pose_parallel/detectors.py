"""Detectors for the YOLOv8 + pose parallel pipeline (self-contained).

  YoloV8Detector -- YOLOv8 object detection (uses the vendored yolov8 package).
  PoseEstimator  -- YOLO11-pose, ported from the DeepStream pose_render_cpu.cpp.

Each exposes .infer(bgr) -> result and .draw(frame, result).
"""

import cv2
import numpy as np
import onnxruntime as ort

from yolov8 import YOLOv8 as _YOLOv8
from yolov8.utils import class_names, colors


def _providers():
    avail = ort.get_available_providers()
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] \
        if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"]


def _letterbox(bgr, size=640, color=114):
    h, w = bgr.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    canvas = np.full((size, size, 3), color, np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(rgb, (nw, nh))
    x = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None]
    return x.astype(np.float32), scale, pad_x, pad_y


# ===========================================================================
# YOLOv8 object detection
# ===========================================================================
class YoloV8Detector:
    def __init__(self, model_path, conf=0.4, iou=0.5, input_size=640):
        self.det = _YOLOv8(model_path, conf_thres=conf, iou_thres=iou)
        if not isinstance(self.det.input_width, int) or not isinstance(self.det.input_height, int):
            self.det.input_width = self.det.input_height = input_size

    def infer(self, bgr):
        boxes, scores, class_ids = self.det(bgr)
        return list(zip(boxes, scores, class_ids))

    @staticmethod
    def draw(frame, dets):
        for box, score, cid in dets:
            x1, y1, x2, y2 = box.astype(int)
            col = tuple(int(c) for c in colors[cid])
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            label = f"{class_names[cid]} {int(score * 100)}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            yb = max(y1, th + 4)
            cv2.rectangle(frame, (x1, yb - th - 4), (x1 + tw, yb), col, -1)
            cv2.putText(frame, label, (x1, yb - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)


# ===========================================================================
# YOLO11-pose
# ===========================================================================
class PoseEstimator:
    KPTS = 17
    CH = 56
    EDGES = [(0, 1), (0, 2), (1, 3), (2, 4),
             (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
             (5, 11), (6, 12), (11, 12),
             (11, 13), (13, 15), (12, 14), (14, 16)]
    LINE_COLOR = (0, 255, 0)
    DOT_COLOR = (0, 0, 255)

    def __init__(self, model_path, conf=0.25, iou=0.45, kconf=0.30, input_size=640):
        self.s = ort.InferenceSession(model_path, providers=_providers())
        self.in_name = self.s.get_inputs()[0].name
        self.out_name = self.s.get_outputs()[0].name
        self.size = input_size
        self.conf, self.iou, self.kconf = conf, iou, kconf

    def infer(self, bgr):
        x, scale, pad_x, pad_y = _letterbox(bgr, self.size)
        o = np.squeeze(self.s.run([self.out_name], {self.in_name: x})[0])   # (56, 8400)
        if o.shape[0] != self.CH:
            return []
        scores = o[4]
        keep = np.where(scores >= self.conf)[0]
        if keep.size == 0:
            return []

        cx, cy = o[0, keep], o[1, keep]
        bw, bh = o[2, keep], o[3, keep]
        fx, fy = (cx - pad_x) / scale, (cy - pad_y) / scale
        fw, fh = bw / scale, bh / scale
        x1, y1, x2, y2 = fx - fw / 2, fy - fh / 2, fx + fw / 2, fy + fh / 2

        kp = o[5:5 + self.KPTS * 3, keep].reshape(self.KPTS, 3, keep.size)
        kx = (kp[:, 0, :] - pad_x) / scale
        ky = (kp[:, 1, :] - pad_y) / scale
        kc = kp[:, 2, :]

        dets = [dict(box=(x1[i], y1[i], x2[i], y2[i]), score=float(scores[keep][i]),
                     kx=kx[:, i], ky=ky[:, i], kc=kc[:, i]) for i in range(keep.size)]
        return self._nms(dets)

    def _nms(self, dets):
        dets.sort(key=lambda d: d["score"], reverse=True)
        keep = []
        for d in dets:
            if all(self._iou(d["box"], k["box"]) <= self.iou for k in keep):
                keep.append(d)
        return keep

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def draw(self, frame, dets):
        for d in dets:
            kx, ky, kc = d["kx"], d["ky"], d["kc"]
            for a, b in self.EDGES:
                if kc[a] >= self.kconf and kc[b] >= self.kconf:
                    cv2.line(frame, (int(kx[a]), int(ky[a])), (int(kx[b]), int(ky[b])),
                             self.LINE_COLOR, 2, cv2.LINE_AA)
            for k in range(self.KPTS):
                if kc[k] >= self.kconf:
                    cv2.circle(frame, (int(kx[k]), int(ky[k])), 4, self.DOT_COLOR, -1, cv2.LINE_AA)


def open_source(spec):
    if spec is not None and str(spec).isdigit():
        return cv2.VideoCapture(int(spec))
    return cv2.VideoCapture(spec)
