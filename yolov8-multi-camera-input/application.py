import cv2
import numpy as np
import onnxruntime as ort

# Model has dynamic input [batch, 3, H, W] and dynamic output [batch, 84, anchors].
# Choose a fixed inference resolution (multiple of 32 for YOLOv8).
INPUT_W, INPUT_H = 640, 640
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5

with open("label.txt", "r") as f:
    CLASSES = [line.strip() for line in f if line.strip()]

camera_names = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
]

caps = [cv2.VideoCapture(f"{name}.mp4") for name in camera_names]

session = ort.InferenceSession(
    "./yolov8n.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name


def preprocess(frames):
    batch = np.empty((len(frames), 3, INPUT_H, INPUT_W), dtype=np.float32)
    for i, frame in enumerate(frames):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (INPUT_W, INPUT_H))
        batch[i] = img.transpose(2, 0, 1)
    batch /= 255.0
    return batch


def postprocess(output, frame, scale_x, scale_y):
    # output shape: [84, anchors] -> [anchors, 84]
    pred = output.T
    class_scores = pred[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_scores)), class_ids]

    keep = confidences > CONF_THRESHOLD
    if not np.any(keep):
        return frame

    pred = pred[keep]
    class_ids = class_ids[keep]
    confidences = confidences[keep]

    # xywh (center) -> xyxy in original frame coords
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x1 = (cx - w / 2) * scale_x
    y1 = (cy - h / 2) * scale_y
    x2 = (cx + w / 2) * scale_x
    y2 = (cy + h / 2) * scale_y

    boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(boxes, confidences.tolist(), CONF_THRESHOLD, IOU_THRESHOLD)

    for i in np.array(indices).flatten():
        bx, by, bw, bh = map(int, boxes[i])
        label = f"{CLASSES[class_ids[i]]} {confidences[i]:.2f}"
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.putText(frame, label, (bx, by - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame


print(f"Running batch inference (batch={len(camera_names)})...")

while True:
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        frames.append(frame)

    batch = preprocess(frames)
    outputs = session.run(None, {input_name: batch})[0]  # [B, 84, anchors]

    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        postprocess(outputs[i], frame, w / INPUT_W, h / INPUT_H)
        cv2.putText(frame, camera_names[i], (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    tiles = [cv2.resize(f, (640, 360)) for f in frames]
    top = np.hstack(tiles[0:3])
    bottom = np.hstack(tiles[3:6])
    cv2.imshow("YOLOv8 Batch Inference - 6 Cameras", np.vstack([top, bottom]))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
