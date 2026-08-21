"""Single-image sanity test for wiwiewei18/smart-shelf-tracker (out-of-stock
shelf detection), v3.

Background on why this took two iterations:
v1 used this model at its "normal" default confidence (0.3) and got zero
detections on real empty-shelf photos. Checking raw scores showed the
model's OOS-class confidence peaks around 0.006-0.03 on real photos -- a
classic small-dataset (497 images) fine-tune symptom: correct spatial
localization, badly miscalibrated (very low) confidence magnitude. This was
confirmed, not assumed: plotting the model's own top-8 lowest-confidence
boxes on the empty Sainsbury's pasta shelf photo showed its #4 box (conf
0.0064) landing EXACTLY on the true empty gap -- the model already "sees"
it, it just never says so with any confidence.

v2 tried swapping to foduucom/product-detection-in-shelf-yolov8 and
inferring gaps from absent "product" detections instead of trusting a
direct "empty" class. That model turned out to be trained on "Retail
Coolers" data (visible in its own ONNX export metadata description path)
-- a refrigerated-drinks domain -- so it barely detected any of the dry
pasta/sauce products in a general shelf photo, making the gap inference
wrong (it flagged a tiny jar-shelf gap while missing the real, much larger
empty shelf entirely).

v3 (this version) goes back to wiwiewei18/smart-shelf-tracker but at a very
low confidence threshold, with two extra filters to suppress the false
positives that appear at that threshold: (1) drop boxes centered in the
bottom `floor_frac` of the image (the model's low-confidence noise floor
includes speckled-floor detections well below shelf height), (2) drop boxes
smaller than `min_area_frac` of the image (tiny noise boxes). Verified on
both a genuinely empty shelf (1 correct gap survives) and a fully-stocked
shelf (0 false positives survive).

Ultralytics YOLO11n detector, 1 class: "100- O-O-S" (raw Roboflow label for
out-of-stock, displayed here as OOS). Input images[1,3,640,640] (RGB,
letterbox to 640, grey (114,114,114) pad, /255, no mean/std). Output
output0[1,5,8400] = 4 bbox (cx,cy,w,h in 640-space) + 1 class score, raw
(nms:False) -> transpose + confidence filter + NMS, per the embedded ONNX
metadata (names, imgsz, task, nms all read from custom_metadata_map, not
guessed).
"""

import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

CLASS_NAMES = ["OOS"]  # raw label in the model: "100- O-O-S" (out of stock)

# ---------------------------------------------------------------------------
# Step 2 — model path resolver (ships verbatim)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIRS = ("model", "models", "weights", "onnx", "checkpoints", ".")
DEFAULT_HF_REPO = None  # ships only .pt on HF; exported model/model.onnx is committed locally

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

    def postprocess(self, outputs, confidence_threshold, nms_threshold, floor_frac, min_area_frac):
        # output0: [1, 5, 8400] -> [8400, 5]; cols 0:4 box (cx,cy,w,h), col 4 class score
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

        rects = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)  # x,y,w,h for cv2 NMS
        idxs = cv2.dnn.NMSBoxes(rects.tolist(), confidences.tolist(),
                                confidence_threshold, nms_threshold)
        idxs = np.array(idxs).flatten() if len(idxs) else []

        dets = []
        for i in idxs:
            area = (x2[i] - x1[i]) * (y2[i] - y1[i])
            cy_i = (y1[i] + y2[i]) / 2
            # this model's low-confidence noise floor is dominated by two
            # patterns: tiny boxes, and boxes on the floor well below shelf
            # height -- both filtered out here rather than trusted blindly
            if cy_i > floor_frac * self.orig_h:
                continue
            if area < min_area_frac * self.orig_w * self.orig_h:
                continue
            dets.append({
                "box": [x1[i], y1[i], x2[i], y2[i]],
                "score": float(confidences[i]),
                "class": int(class_ids[i]),
            })
        return dets

    def draw_detections(self, image_path, detections, output_path):
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        # scale text/line thickness to image resolution so labels stay legible
        scale = max(w, h) / 1400
        box_thick = max(3, int(6 * scale))
        font_scale = max(1.2, 1.7 * scale)
        font_thick = max(2, int(3 * scale))

        for det in detections:
            x1, y1, x2, y2 = map(int, det["box"])
            color = (0, 0, 255)  # red = out-of-stock alert
            cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thick)
            label = f'OOS {det["score"]:.3f}'
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            label_y = max(y1, th + baseline + 10)
            cv2.rectangle(image, (x1, label_y - th - baseline - 10), (x1 + tw + 16, label_y), color, -1)
            cv2.putText(image, label, (x1 + 8, label_y - baseline - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

        banner = f"OUT-OF-STOCK ALERT: {len(detections)}" if detections else "SHELF OK: 0 out-of-stock gaps"
        banner_scale = max(1.6, 2.2 * scale)
        banner_thick = max(3, int(4 * scale))
        (tw, th), baseline = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, banner_scale, banner_thick)
        pad = int(16 * scale) + 10
        cv2.rectangle(image, (0, 0), (min(w, tw + pad * 2), th + baseline + pad * 2), (0, 0, 0), -1)
        color = (0, 0, 255) if detections else (0, 255, 0)
        cv2.putText(image, banner, (pad, th + pad),
                    cv2.FONT_HERSHEY_SIMPLEX, banner_scale, color, banner_thick, cv2.LINE_AA)
        cv2.imwrite(output_path, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-image sanity test for the smart-shelf-tracker (out-of-stock) ONNX model.")
    parser.add_argument("-m", "--model", default=None,
                        help="Path to .onnx, a folder containing one, or a HF ref. "
                             "Omit to auto-discover in: " + ", ".join(DEFAULT_MODEL_DIRS))
    parser.add_argument("-i", "--image", required=True)
    parser.add_argument("-o", "--output", default="output.jpg")
    parser.add_argument("-c", "--conf-threshold", type=float, default=0.003,
                         help="Low by design -- this checkpoint's OOS confidence is badly "
                              "miscalibrated (correct location, ~0.006-0.03 typical peak score)")
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--floor-frac", type=float, default=0.85,
                         help="Ignore detections centered below this fraction of image height (floor)")
    parser.add_argument("--min-area-frac", type=float, default=0.01,
                         help="Ignore detections smaller than this fraction of image area (noise)")
    args = parser.parse_args()

    model = ModelTester(resolve_model_path(args.model))
    print(f"[input ] {model.input_name} {model.input_shape}")
    print(f"[output] {model.output_names}")

    outputs = model.run_inference(args.image)
    print(f"[raw   ] {outputs[0].shape}  min={outputs[0].min():.4f} max={outputs[0].max():.4f}")
    dets = model.postprocess(outputs, args.conf_threshold, args.nms_threshold,
                              args.floor_frac, args.min_area_frac)
    print(f"[detect] {len(dets)} out-of-stock section(s)")
    for d in dets:
        print(f"         OOS  {d['score']:.4f}  "
              f"[{d['box'][0]:.0f},{d['box'][1]:.0f},{d['box'][2]:.0f},{d['box'][3]:.0f}]")
    model.draw_detections(args.image, dets, args.output)
    print(f"[done  ] wrote {args.output}")
