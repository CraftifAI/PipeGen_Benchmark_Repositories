"""YOLOv11 instance segmentation (ONNX).

Copied near-verbatim from the source repo
`yolo-detection-and-segmentation/yolov11_seg/YOLOv11Seg.py` so this folder is
self-contained. A per-frame inference timer (`infer_ms`) was added for the HUD.

API:
    seg = YOLOv11Seg(model_path, coco_label_txt_path, bbox_threshold=0.6, nms_iou_threshold=0.4)
    annotated640 = seg.segment(frame)   # 640x640 BGR image with masks drawn
"""

import time

import cv2
import numpy as np
import onnxruntime as ort


class YOLOv11Seg:

    def __init__(
        self,
        model_path,
        coco_label,
        nms_iou_threshold=0.4,
        thickness=1,
        font_scale=0.5,
        bbox_threshold=0.6,
        mask_threshold=0.5,
        alpha=0.5,
        height=640,
        width=640,
    ):

        self.model_path = model_path
        self.coco_label = coco_label

        self.thickness = thickness
        self.font_scale = font_scale
        self.bbox_threshold = bbox_threshold
        self.mask_threshold = mask_threshold
        self.alpha = alpha
        self.height = height
        self.width = width
        self.nms_iou_threshold = nms_iou_threshold

        with open(self.coco_label, "r") as f:
            self.coco = [line.strip() for line in f.readlines()]

        self.session = ort.InferenceSession(
            self.model_path,
            providers=ort.get_available_providers(),
        )

        self.input_name = self.session.get_inputs()[0].name
        self.infer_ms = 0.0

    # --------------------------------------------------------

    def __call__(self, frame):
        return self.segment(frame)

    # --------------------------------------------------------

    def preprocess(self, frame):

        frame_resized = cv2.resize(frame, (self.width, self.height))

        tensor = frame_resized.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        return frame_resized, tensor

    # --------------------------------------------------------

    def inference(self, tensor):

        t0 = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: tensor})
        self.infer_ms = (time.perf_counter() - t0) * 1000.0

        seg_det = outputs[0]      # [1,116,8400]
        seg_mask = outputs[1][0]  # [32,160,160]

        return seg_det, seg_mask

    # --------------------------------------------------------

    def segment(self, frame):
        """Run segmentation on a frame. Returns the resized frame with masks drawn."""
        frame_resized, tensor = self.preprocess(frame)
        det, mask = self.inference(tensor)
        return self.postprocess(frame_resized, det, mask)

    # --------------------------------------------------------

    def postprocess(self, frame, det_frame, mask_frame):

        det_frame = np.transpose(det_frame[0], (1, 0))  # 8400,116

        boxes = []
        confs = []
        class_ids = []
        mask_weights_list = []
        xyxy_list = []

        for det in det_frame:

            x, y, w, h = det[0:4]
            class_logits = det[4:84]
            mask_weights = det[84:116]

            class_scores = 1 / (1 + np.exp(-class_logits))

            class_id = np.argmax(class_scores)
            conf = class_scores[class_id]

            if conf < self.bbox_threshold:
                continue

            x_min = int(x - w / 2)
            y_min = int(y - h / 2)
            x_max = int(x + w / 2)
            y_max = int(y + h / 2)

            if not (
                x_min >= 0
                and y_min >= 0
                and x_max <= self.width
                and y_max <= self.height
                and x_max > x_min
                and y_max > y_min
            ):
                continue

            boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
            confs.append(float(conf))
            class_ids.append(class_id)
            mask_weights_list.append(mask_weights)
            xyxy_list.append((x_min, y_min, x_max, y_max))

        indices = cv2.dnn.NMSBoxes(
            boxes,
            confs,
            self.bbox_threshold,
            self.nms_iou_threshold,
        )

        if len(indices) == 0:
            return frame

        for i in indices.flatten():

            x_min, y_min, x_max, y_max = xyxy_list[i]
            class_id = class_ids[i]
            conf = confs[i]
            label = self.coco[class_id]

            color = np.array(
                [
                    (37 * class_id) % 255,
                    (17 * class_id) % 255,
                    (29 * class_id) % 255,
                ],
                dtype=np.uint8,
            )

            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                color.tolist(),
                self.thickness,
            )

            cv2.putText(
                frame,
                f"{label}: {conf:.2f}",
                (x_min, max(y_min - 2, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                color.tolist(),
                self.thickness,
            )

            mask_weights = mask_weights_list[i]

            mask_low_res = np.tensordot(
                mask_weights,
                mask_frame,
                axes=(0, 0),
            )

            scale_x1 = int(x_min * mask_frame.shape[2] / self.width)
            scale_y1 = int(y_min * mask_frame.shape[1] / self.height)
            scale_x2 = int(x_max * mask_frame.shape[2] / self.width)
            scale_y2 = int(y_max * mask_frame.shape[1] / self.height)

            mask_cropped = mask_low_res[
                scale_y1:scale_y2,
                scale_x1:scale_x2,
            ]

            if mask_cropped.size == 0:
                continue

            mask_resized = cv2.resize(
                mask_cropped,
                (x_max - x_min, y_max - y_min),
                interpolation=cv2.INTER_CUBIC,
            )

            mask_resized = 1 / (1 + np.exp(-mask_resized))
            mask_bin = mask_resized > self.mask_threshold

            frame_roi = frame[y_min:y_max, x_min:x_max]

            frame_roi[mask_bin] = (
                (1 - self.alpha) * frame_roi[mask_bin]
                + self.alpha * color
            ).astype(np.uint8)

        return frame
