"""Single-image sanity test for vehicle detection.

No GitHub repo or existing model file for this use case -- but unlike the
recent niche tasks (smoking, out-of-stock shelves, container fill level),
vehicle detection is a standard, well-covered COCO category, so no custom
fine-tune search was needed. Uses Ultralytics' own official YOLO11s
checkpoint (Ultralytics/YOLO11 on Hugging Face), COCO-pretrained (80
classes). Only ships .pt, so exported once with
`YOLO("yolo11s.pt").export(format="onnx", opset=12, simplify=True)` (a
mechanical export, not training) -> model/yolo11s.onnx.

Input images[1,3,640,640] (RGB, letterbox to 640 with grey (114,114,114)
padding, /255, no mean/std). Output output0[1,84,8400] = 4 bbox
(cx,cy,w,h in 640-space) + 80 class scores, raw (export used nms:False)
-> needs transpose + confidence filter + NMS.

"Vehicle" here means the COCO classes bicycle, car, motorcycle, bus, train,
truck, boat -- everything else is detected too (for a complete sanity
check) but only vehicle classes count toward the vehicle-count banner.
"""

import argparse
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
]  # COCO-80, from Ultralytics YOLO11 model.names

VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "train", "truck", "boat"}

# ---------------------------------------------------------------------------
# Step 2 — model path resolver (ships verbatim)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIRS = ("model", "models", "weights", "onnx", "checkpoints", ".")
# Ultralytics/YOLO11 ships only .pt files (no .onnx), so auto-fetch can't
# resolve it directly -- the exported model/yolo11s.onnx is committed locally.
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
            ckpts = [f for f in files if f.endswith((".pt", ".pth", ".safetensors"))]
            raise FileNotFoundError(
                f"{repo_id} has no .onnx. Checkpoints present: {ckpts or 'none'}\n"
                "Export to ONNX first."
            )
        if len(onnx_files) > 1:
            raise RuntimeError(
                f"{repo_id} has {len(onnx_files)} .onnx files — pick one with\n  "
                + "\n  ".join(f"-m {repo_id}/blob/{revision}/{f}" for f in onnx_files)
            )
        filename = onnx_files[0]

    path = hf_hub_download(repo_id, filename, revision=revision,
                           local_dir=local_dir, token=token)
    for sidecar in (f"{filename}.data", f"{filename}_data", f"{filename}.data_0"):
        if sidecar in files:
            hf_hub_download(repo_id, sidecar, revision=revision,
                            local_dir=local_dir, token=token)
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

    if not candidates:
        if DEFAULT_HF_REPO:
            return fetch_from_hf(DEFAULT_HF_REPO)
        raise FileNotFoundError(
            "No .onnx found in: " + ", ".join(str(d) for d in search_dirs)
        )
    if len(candidates) > 1:
        listing = "\n  ".join(f"{c}  ({c.stat().st_size / 1e6:.1f} MB)" for c in candidates)
        raise RuntimeError(f"Found {len(candidates)} .onnx files — pick one with -m:\n  {listing}")

    print(f"[model] auto-resolved: {candidates[0]}")
    return str(candidates[0])


# ---------------------------------------------------------------------------
class ModelTester:
    def __init__(self, model_path):
        self.model_path = model_path
        self.ort_session = ort.InferenceSession(
            model_path, providers=ort.get_available_providers()
        )
        inp = self.ort_session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape          # [1, 3, 640, 640]
        self.input_h, self.input_w = inp.shape[2], inp.shape[3]
        self.output_names = [o.name for o in self.ort_session.get_outputs()]

    def preprocess_image(self, image_path):
        """YOLO letterbox: RGB, aspect-preserving resize to 640, grey pad, /255."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"could not read image: {image_path}")
        self.orig_h, self.orig_w = img.shape[:2]

        r = min(self.input_h / self.orig_h, self.input_w / self.orig_w)
        new_w, new_h = round(self.orig_w * r), round(self.orig_h * r)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        pad_x, pad_y = (self.input_w - new_w) // 2, (self.input_h - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        self.ratio, self.pad_x, self.pad_y = r, pad_x, pad_y

        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]  # HWC -> NCHW
        return np.ascontiguousarray(blob)

    def run_inference(self, image_path):
        blob = self.preprocess_image(image_path)
        return self.ort_session.run(self.output_names, {self.input_name: blob})

    def postprocess(self, outputs, confidence_threshold=0.3, nms_threshold=0.45):
        # output0: [1, 84, 8400] -> [8400, 84]; cols 0:4 box (cx,cy,w,h), 4:84 class scores
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
                    "class": int(class_ids[oi]),
                })
        return dets

    def draw_detections(self, image_path, detections, output_path):
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        scale = max(w, h) / 1400
        box_thick = max(2, int(4 * scale))
        label_scale = max(0.9, 1.2 * scale)
        label_thick = max(2, int(3 * scale))

        vehicle_count = 0
        for det in detections:
            x1, y1, x2, y2 = map(int, det["box"])
            name = CLASS_NAMES[det["class"]]
            is_vehicle = name in VEHICLE_CLASSES
            vehicle_count += int(is_vehicle)
            color = (0, 200, 0) if is_vehicle else (160, 160, 160)
            label = f'{name} {det["score"]:.2f}'
            cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thick)
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thick)
            label_y = max(y1, th + baseline + 8)
            cv2.rectangle(image, (x1, label_y - th - baseline - 8), (x1 + tw + 12, label_y), color, -1)
            cv2.putText(image, label, (x1 + 6, label_y - baseline - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, label_scale, (0, 0, 0), label_thick, cv2.LINE_AA)

        banner = f"VEHICLES DETECTED: {vehicle_count}"
        banner_scale = max(1.6, 2.1 * scale)
        banner_thick = max(3, int(4 * scale))
        (tw, th), baseline = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, banner_scale, banner_thick)
        pad = int(16 * scale) + 10
        cv2.rectangle(image, (0, 0), (min(w, tw + pad * 2), th + baseline + pad * 2), (0, 0, 0), -1)
        cv2.putText(image, banner, (pad, th + pad),
                    cv2.FONT_HERSHEY_SIMPLEX, banner_scale, (0, 255, 0), banner_thick, cv2.LINE_AA)
        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)
        if not cv2.imwrite(output_path, image):
            raise IOError(f"failed to write output image: {output_path}")
        return vehicle_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-image sanity test for vehicle detection (YOLO11s, COCO).")
    parser.add_argument("-m", "--model", default=None,
                        help="Path to .onnx, a folder containing one, or a HF ref. "
                             "Omit to auto-discover in: " + ", ".join(DEFAULT_MODEL_DIRS))
    parser.add_argument("-i", "--image", required=True)
    parser.add_argument("-o", "--output", default="output.jpg")
    parser.add_argument("-c", "--conf-threshold", type=float, default=0.3)
    args = parser.parse_args()

    model = ModelTester(resolve_model_path(args.model))
    print(f"[input ] {model.input_name} {model.input_shape}")
    print(f"[output] {model.output_names}")

    outputs = model.run_inference(args.image)
    print(f"[raw   ] {outputs[0].shape}  min={outputs[0].min():.4f} max={outputs[0].max():.4f}")
    dets = model.postprocess(outputs, confidence_threshold=args.conf_threshold)
    print(f"[detect] {len(dets)} objects")
    for d in dets:
        print(f"         {CLASS_NAMES[d['class']]:<14} {d['score']:.3f}  "
              f"[{d['box'][0]:.0f},{d['box'][1]:.0f},{d['box'][2]:.0f},{d['box'][3]:.0f}]")
    vehicle_count = model.draw_detections(args.image, dets, args.output)
    print(f"[result] {vehicle_count} vehicle(s)")
    print(f"[done  ] wrote {args.output}")
