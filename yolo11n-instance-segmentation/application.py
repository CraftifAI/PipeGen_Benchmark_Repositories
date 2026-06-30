import os
import cv2
import argparse
import numpy as np
import onnxruntime as ort


class InstanceSegmentation:
    def __init__(
        self,
        input_video,
        output_video,
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

        self.input_video = input_video
        self.output_video = output_video
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

        # load labels once
        with open(self.coco_label, "r") as f:
            self.coco = [line.strip() for line in f.readlines()]

        # load ONNX session once
        self.session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name

    # --------------------------------------------------------

    def preprocess(self, frame):

        frame = cv2.resize(frame, (self.width, self.height))

        tensor = frame.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        return frame, tensor

    # --------------------------------------------------------

    def inference(self, tensor):

        outputs = self.session.run(None, {self.input_name: tensor})

        seg_det = outputs[0]      # [1,116,8400]
        seg_mask = outputs[1][0]  # [32,160,160]

        return seg_det, seg_mask

    # --------------------------------------------------------

    def postprocess(self, frame, det_frame, mask_frame):

        det_frame = np.transpose(det_frame[0], (1, 0))  # 8400,116

        boxes = []
        confs = []
        class_ids = []
        mask_weights_list = []
        xyxy_list = []

        # decode detections
        for det in det_frame:

            x, y, w, h = det[0:4]
            class_logits = det[4:84]
            mask_weights = det[84:116]

            # sigmoid
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

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes,
            confs,
            self.bbox_threshold,
            self.nms_iou_threshold,
        )

        if len(indices) == 0:
            return frame

        # draw detections
        for i in indices.flatten():

            x_min, y_min, x_max, y_max = xyxy_list[i]
            class_id = class_ids[i]
            conf = confs[i]
            label = self.coco[class_id]

            # class based color
            color = np.array(
                [
                    (37 * class_id) % 255,
                    (17 * class_id) % 255,
                    (29 * class_id) % 255,
                ],
                dtype=np.uint8,
            )

            # draw bbox
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

            # mask generation
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

    # --------------------------------------------------------

    def run(self):

        cap = cv2.VideoCapture(self.input_video)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            self.output_video,
            fourcc,
            20.0,
            (self.width, self.height),
        )

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            frame, tensor = self.preprocess(frame)

            det, mask = self.inference(tensor)

            frame = self.postprocess(frame, det, mask)

            cv2.imshow("output", frame)

            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()


# ------------------------------------------------------------

if __name__ == "__main__":

    dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_video",
        default="/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4",
    )

    parser.add_argument(
        "-o",
        "--output_video",
        default=os.path.join(dir, "output.mp4"),
    )

    parser.add_argument(
        "-m",
        "--model",
        default=os.path.join(dir, "models", "yolo11n-seg.onnx"),
    )

    parser.add_argument(
        "-l",
        "--coco_label",
        default=os.path.join(dir, "coco.txt"),
    )

    args = parser.parse_args()

    seg = InstanceSegmentation(
        args.input_video,
        args.output_video,
        args.model,
        args.coco_label,
    )

    seg.run()
