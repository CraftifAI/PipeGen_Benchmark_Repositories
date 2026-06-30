import cv2
import numpy as np
import onnxruntime as ort

with open("labels.txt") as f:
    COCO_CLASSES = [line.strip() for line in f if line.strip()]

# Generate distinct colors for each class
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(134, 3), dtype=np.uint8)

INPUT_SIZE = (384, 384)
SCORE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5


def preprocess(frame):
    img = cv2.resize(frame, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 384, 384)
    return img


def postprocess(class_logits, mask_logits, orig_h, orig_w):
    """
    class_logits: (1, 100, 134)  — per-query class scores
    mask_logits:  (1, 100, 96, 96) — per-query soft masks
    """
    class_logits = class_logits[0]   # (100, 134)
    mask_logits  = mask_logits[0]    # (100, 96, 96)

    # Softmax over class dimension; last class is typically "no-object"
    scores_all = np.exp(class_logits - class_logits.max(axis=1, keepdims=True))
    scores_all /= scores_all.sum(axis=1, keepdims=True)

    # Best class per query (exclude last "no-object" class index 133)
    best_class = np.argmax(scores_all[:, :-1], axis=1)   # (100,)
    best_score = scores_all[np.arange(100), best_class]  # (100,)

    keep = best_score >= SCORE_THRESHOLD
    if not keep.any():
        return []

    instances = []
    for idx in np.where(keep)[0]:
        cls_id = int(best_class[idx])
        score  = float(best_score[idx])
        mask   = mask_logits[idx]                  # (96, 96)
        mask   = 1 / (1 + np.exp(-mask))           # sigmoid
        mask   = cv2.resize(mask, (orig_w, orig_h))
        binary = (mask >= MASK_THRESHOLD).astype(np.uint8)
        instances.append((cls_id, score, binary))

    # Sort by score descending
    instances.sort(key=lambda x: x[1], reverse=True)
    return instances


def draw_instances(frame, instances):
    overlay = frame.copy()
    for cls_id, score, binary in instances:
        color = COLORS[cls_id % len(COLORS)].tolist()
        overlay[binary == 1] = (
            overlay[binary == 1] * 0.5 + np.array(color) * 0.5
        ).astype(np.uint8)

        # Draw contour
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

        # Label at the bounding box top-left
        ys, xs = np.where(binary)
        if len(xs) == 0:
            continue
        x1, y1 = int(xs.min()), int(ys.min())
        label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"cls{cls_id}"
        label_text = f"{label} {score:.2f}"
        cv2.rectangle(overlay, (x1, y1 - 18), (x1 + len(label_text) * 9, y1), color, -1)
        cv2.putText(overlay, label_text, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay


def main():
    session = ort.InferenceSession(
        "models/Mask2Former.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name   # "image"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    print("Running Mask2Former instance segmentation — press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w = frame.shape[:2]
        blob = preprocess(frame)

        outputs = session.run(None, {input_name: blob})
        # outputs[0] -> class_idx (1,100,134), outputs[1] -> masks (1,100,96,96)
        class_logits = outputs[0]
        mask_logits  = outputs[1]

        instances = postprocess(class_logits, mask_logits, orig_h, orig_w)
        vis = draw_instances(frame, instances)

        cv2.putText(vis, f"Instances: {len(instances)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Mask2Former Instance Segmentation", vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
