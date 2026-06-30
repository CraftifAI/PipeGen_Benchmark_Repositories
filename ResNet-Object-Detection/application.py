import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "models/resnet18_trafficcamnet_pruned_bs1.onnx"
LABELS_PATH = "labels.txt"
INPUT_W, INPUT_H = 960, 544
COVERAGE_THRESHOLD = 0.4
NMS_IOU_THRESHOLD = 0.3
STRIDE = 16  # 960/60 = 544/34 = 16

NUM_CLASSES = 4


def load_labels(path):
    with open(path) as f:
        labels = {i: line.strip() for i, line in enumerate(f) if line.strip()}
    print(f"Loaded labels: {labels}")
    return labels


def preprocess(frame):
    img = cv2.resize(frame, (INPUT_W, INPUT_H))
    img = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, normalize
    img = np.transpose(img, (2, 0, 1))                 # HWC -> CHW
    return np.expand_dims(img, axis=0)


def decode_detections(coverage, bboxes, scale_x, scale_y):
    # coverage: [1, 4, 34, 60]
    # bboxes:   [1, 16, 34, 60]  (4 coords per class: x1, y1, x2, y2)
    detections = []

    for c in range(NUM_CLASSES):
        cov = coverage[0, c]          # [34, 60]
        bx1 = bboxes[0, c * 4 + 0]
        by1 = bboxes[0, c * 4 + 1]
        bx2 = bboxes[0, c * 4 + 2]
        by2 = bboxes[0, c * 4 + 3]

        rows, cols = np.where(cov > COVERAGE_THRESHOLD)
        for row, col in zip(rows, cols):
            score = float(cov[row, col])
            # DetectNet v2: bbox channels are (left, top, right, bottom) offsets
            # from the grid cell center, normalized by stride
            cx = col * STRIDE + STRIDE / 2
            cy = row * STRIDE + STRIDE / 2
            x1 = (cx - float(bx1[row, col]) * STRIDE) * scale_x
            y1 = (cy - float(by1[row, col]) * STRIDE) * scale_y
            x2 = (cx + float(bx2[row, col]) * STRIDE) * scale_x
            y2 = (cy + float(by2[row, col]) * STRIDE) * scale_y
            if x2 > x1 and y2 > y1:
                detections.append((x1, y1, x2, y2, score, c))

    return detections


def nms(detections):
    if not detections:
        return []

    boxes = np.array([[d[0], d[1], d[2], d[3]] for d in detections], dtype=np.float32)
    scores = np.array([d[4] for d in detections], dtype=np.float32)
    class_ids = np.array([d[5] for d in detections])

    kept = []
    for c in range(NUM_CLASSES):
        idx = np.where(class_ids == c)[0]
        if len(idx) == 0:
            continue
        c_boxes = boxes[idx]
        c_scores = scores[idx]
        rects = [(b[0], b[1], b[2] - b[0], b[3] - b[1]) for b in c_boxes]
        keep = cv2.dnn.NMSBoxes(rects, c_scores.tolist(), COVERAGE_THRESHOLD, NMS_IOU_THRESHOLD)
        if len(keep) > 0:
            for i in keep.flatten():
                kept.append(detections[idx[i]])

    return kept


COLORS = [
    (0, 200, 0),    # car - green
    (255, 128, 0),  # bicycle - orange
    (0, 128, 255),  # person - blue
    (0, 0, 220),    # road_sign - red
]


def draw(frame, detections, labels):
    for x1, y1, x2, y2, score, class_id in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = COLORS[class_id % len(COLORS)]
        label = labels.get(class_id, str(class_id))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(frame, text, (x1 + 1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def main():
    labels = load_labels(LABELS_PATH)

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    out_cov_name = "output_cov/Sigmoid:0"
    out_bbox_name = "output_bbox/BiasAdd:0"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    print("Running — press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w = frame.shape[:2]
        scale_x = orig_w / INPUT_W
        scale_y = orig_h / INPUT_H

        inp = preprocess(frame)
        coverage, bboxes = session.run([out_cov_name, out_bbox_name], {input_name: inp})

        print(f"coverage max={coverage.max():.4f}  min={coverage.min():.4f}  "
              f"bbox max={bboxes.max():.1f}  min={bboxes.min():.1f}")

        detections = decode_detections(coverage, bboxes, scale_x, scale_y)
        detections = nms(detections)
        frame = draw(frame, detections, labels)

        cv2.imshow("TrafficCamNet Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
