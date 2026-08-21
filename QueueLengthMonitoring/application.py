"""Single-image sanity test for queue length monitoring.

No dedicated "queue monitoring" model exists (queue length is a counting +
zone-membership rule on top of person detection, not a trained visual
class of its own -- same situation as Dock Utilization/Shelf Occupancy).
Reuses Ultralytics YOLO11s (COCO-pretrained, already verified in
VehicleDetection/DockUtilization/QueueLengthMonitoring's sibling use
cases), filtered to the "person" class, plus a configurable queue-zone
polygon.

Queue length here means "how many people are inside the queue zone" --
the standard actionable metric real queue-monitoring systems report
(feeds directly into estimated-wait-time = count * average service time).
It does not attempt to determine queue ORDER or detect a physical queue
line shape (e.g. serpentine stanchion queues) -- that would need either a
zone shaped to match the actual stanchion layout (supported via
--zone-file) or a dedicated line-fitting step this script doesn't do.

Detector I/O: identical to VehicleDetection/DockUtilization (input
images[1,3,640,640] RGB letterboxed, /255; output [1,84,8400] raw
4-box+80-class, needs transpose + confidence filter + NMS).
"""

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]  # COCO-80

PERSON_CLASS = "person"

# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIRS = ("model", "models", "weights", "onnx", "checkpoints", ".")
DEFAULT_HF_REPO = None

HF_REF = re.compile(
    r"^(?:https?://huggingface\.co/)?"
    r"(?P<repo>[\w.-]+/[\w.-]+)"
    r"(?:/(?:blob|resolve)/(?P<rev>[^/]+)/(?P<file>.+?))?/?$"
)


def looks_like_hf_ref(value):
    if Path(value).expanduser().exists():
        return False
    if value.startswith(("http://", "https://")):
        return "huggingface.co/" in value
    if value.startswith((".", "/", "~")):
        return False
    if value.endswith(".onnx") and "/blob/" not in value and "/resolve/" not in value:
        return False
    return bool(HF_REF.match(value))


def fetch_from_hf(ref, local_dir="model"):
    from huggingface_hub import hf_hub_download, list_repo_files

    m = HF_REF.match(ref)
    repo_id, revision, filename = m.group("repo"), m.group("rev") or "main", m.group("file")
    token = os.environ.get("HF_TOKEN")
    files = list_repo_files(repo_id, revision=revision, token=token)
    if filename is None:
        onnx_files = [f for f in files if f.endswith(".onnx")]
        if not onnx_files:
            raise FileNotFoundError(f"{repo_id} has no .onnx.")
        if len(onnx_files) > 1:
            raise RuntimeError(f"{repo_id} has {len(onnx_files)} .onnx files — pick one with -m")
        filename = onnx_files[0]
    path = hf_hub_download(repo_id, filename, revision=revision, local_dir=local_dir, token=token)
    print(f"[model] fetched {repo_id}:{filename} -> {path}")
    return path


def resolve_model_path(explicit=None, search_dirs=DEFAULT_MODEL_DIRS):
    if explicit:
        if looks_like_hf_ref(explicit):
            return fetch_from_hf(explicit)
        p = Path(explicit).expanduser()
        if p.is_file():
            return str(p)
        if p.is_dir():
            search_dirs = (p,)
        else:
            raise FileNotFoundError(f"--model path does not exist: {p}")

    seen, candidates = set(), []
    for d in search_dirs:
        d = Path(d).expanduser()
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*.onnx")):
            if any(part.startswith(".") for part in f.parts):
                continue
            rp = f.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            candidates.append(f)
        if candidates:
            break
    if not candidates:
        if DEFAULT_HF_REPO:
            return fetch_from_hf(DEFAULT_HF_REPO)
        raise FileNotFoundError("No .onnx found in: " + ", ".join(str(d) for d in search_dirs))
    if len(candidates) > 1:
        listing = "\n  ".join(f"{c}  ({c.stat().st_size / 1e6:.1f} MB)" for c in candidates)
        raise RuntimeError(f"Found {len(candidates)} .onnx files — pick one with -m:\n  {listing}")
    print(f"[model] auto-resolved: {candidates[0]}")
    return str(candidates[0])


# ---------------------------------------------------------------------------
class PersonDetector:
    def __init__(self, model_path):
        self.sess = ort.InferenceSession(model_path, providers=ort.get_available_providers())
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.input_h, self.input_w = inp.shape[2], inp.shape[3]
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def preprocess(self, img):
        self.orig_h, self.orig_w = img.shape[:2]
        r = min(self.input_h / self.orig_h, self.input_w / self.orig_w)
        new_w, new_h = round(self.orig_w * r), round(self.orig_h * r)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        pad_x, pad_y = (self.input_w - new_w) // 2, (self.input_h - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        self.ratio, self.pad_x, self.pad_y = r, pad_x, pad_y
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.ascontiguousarray(np.transpose(blob, (2, 0, 1))[None])

    def detect(self, img, confidence_threshold=0.35, nms_threshold=0.45):
        blob = self.preprocess(img)
        outputs = self.sess.run(self.output_names, {self.input_name: blob})
        preds = np.squeeze(outputs[0], 0).T
        boxes_xywh, scores = preds[:, :4], preds[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(scores.shape[0]), class_ids]
        keep = confidences >= confidence_threshold
        boxes_xywh, confidences, class_ids = boxes_xywh[keep], confidences[keep], class_ids[keep]
        if len(boxes_xywh) == 0:
            return []
        cx, cy, w, h = boxes_xywh.T
        x1 = (cx - w / 2 - self.pad_x) / self.ratio
        y1 = (cy - h / 2 - self.pad_y) / self.ratio
        x2 = (cx + w / 2 - self.pad_x) / self.ratio
        y2 = (cy + h / 2 - self.pad_y) / self.ratio
        x1 = np.clip(x1, 0, self.orig_w); x2 = np.clip(x2, 0, self.orig_w)
        y1 = np.clip(y1, 0, self.orig_h); y2 = np.clip(y2, 0, self.orig_h)

        dets = []
        for c in np.unique(class_ids):
            idxs = np.where(class_ids == c)[0]
            rects = np.stack([x1[idxs], y1[idxs], (x2 - x1)[idxs], (y2 - y1)[idxs]], axis=1)
            keep_idx = cv2.dnn.NMSBoxes(rects.tolist(), confidences[idxs].tolist(),
                                        confidence_threshold, nms_threshold)
            keep_idx = np.array(keep_idx).flatten() if len(keep_idx) else []
            for i in keep_idx:
                oi = idxs[i]
                dets.append({
                    "box": [x1[oi], y1[oi], x2[oi], y2[oi]],
                    "score": float(confidences[oi]),
                    "class": CLASS_NAMES[c],
                })
        return dets


# ---------------------------------------------------------------------------
def default_zone(img_shape):
    """Whole frame -- used when no --zone-file is given. A real deployment
    would draw a zone matching the actual queue area/stanchion layout once
    for a fixed camera."""
    h, w = img_shape[:2]
    return [[0, 0], [w, 0], [w, h], [0, h]]


def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.float32), point, False) >= 0


def foot_point(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, y2)  # bottom-center of the box approximates where the person stands


def filter_in_zone(people, zone):
    in_zone = []
    for p in people:
        if point_in_polygon(foot_point(p["box"]), zone):
            in_zone.append(p)
    return in_zone


# ---------------------------------------------------------------------------
def draw_results(image, all_people, in_zone_people, zone, output_path):
    out = image.copy()
    h, w = out.shape[:2]
    scale = max(w, h) / 1400
    box_thick = max(2, int(3 * scale))
    label_scale = max(0.7, 1.0 * scale)
    label_thick = max(1, int(2 * scale))

    overlay = out.copy()
    cv2.fillPoly(overlay, [np.array(zone, dtype=np.int32)], (255, 0, 255))
    out = cv2.addWeighted(overlay, 0.12, out, 0.88, 0)
    cv2.polylines(out, [np.array(zone, dtype=np.int32)], True, (255, 0, 255), max(2, int(3 * scale)))

    in_zone_ids = {id(p) for p in in_zone_people}
    for p in all_people:
        x1, y1, x2, y2 = map(int, p["box"])
        in_q = id(p) in in_zone_ids
        color = (0, 0, 255) if in_q else (150, 150, 150)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thick)
        fx, fy = map(int, foot_point(p["box"]))
        cv2.circle(out, (fx, fy), max(3, int(5 * scale)), color, -1)
        label = f'{p["score"]:.2f}'
        cv2.putText(out, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    label_scale * 0.7, color, max(1, label_thick - 1), cv2.LINE_AA)

    banner = f"QUEUE LENGTH: {len(in_zone_people)} people"
    banner_scale = max(1.4, 1.8 * scale)
    banner_thick = max(2, int(4 * scale))
    (tw, th), baseline = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, banner_scale, banner_thick)
    pad = int(12 * scale) + 8
    cv2.rectangle(out, (0, 0), (min(w, tw + pad * 2), th + baseline + pad * 2), (0, 0, 0), -1)
    color = (0, 0, 255) if len(in_zone_people) >= 8 else (0, 200, 255) if len(in_zone_people) >= 4 else (0, 255, 0)
    cv2.putText(out, banner, (pad, th + pad), cv2.FONT_HERSHEY_SIMPLEX, banner_scale, color, banner_thick, cv2.LINE_AA)
    cv2.imwrite(output_path, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Queue length monitoring sanity test.")
    parser.add_argument("-m", "--model", default=None,
                        help="Path to .onnx, a folder containing one, or a HF ref. "
                             "Omit to auto-discover in: " + ", ".join(DEFAULT_MODEL_DIRS))
    parser.add_argument("-i", "--image", required=True)
    parser.add_argument("-o", "--output", default="output.jpg")
    parser.add_argument("-c", "--conf-threshold", type=float, default=0.35)
    parser.add_argument("--zone-file", default=None,
                         help="JSON file: list of [x,y] polygon points marking the queue area. "
                              "Omit to use the whole frame.")
    args = parser.parse_args()

    model = PersonDetector(resolve_model_path(args.model))
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)

    detections = model.detect(image, confidence_threshold=args.conf_threshold)
    people = [d for d in detections if d["class"] == PERSON_CLASS]
    print(f"[detect] {len(detections)} object(s), {len(people)} person(s)")

    if args.zone_file:
        zone = json.load(open(args.zone_file))
        print(f"[zone  ] loaded from {args.zone_file}")
    else:
        zone = default_zone(image.shape)
        print("[zone  ] no --zone-file given, using whole frame")

    in_zone = filter_in_zone(people, zone)
    print(f"[result] queue length = {len(in_zone)} people in zone (of {len(people)} total detected)")
    for p in in_zone:
        print(f"  person conf={p['score']:.2f}  box={[round(float(v), 1) for v in p['box']]}")

    draw_results(image, people, in_zone, zone, args.output)
    print(f"[done  ] wrote {args.output}")
