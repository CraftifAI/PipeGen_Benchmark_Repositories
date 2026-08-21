"""Single-image sanity test for face mask compliance detection.

Searched Hugging Face/GitHub for a dedicated face-mask detector. Two real
options surfaced, both trained on the same classic Kaggle "Face Mask
Detection" dataset (853 images, 3 classes: with_mask/without_mask/
mask_weared_incorrect):
  - nickmuchi/yolos-small-finetuned-masks -- a HF Transformers YOLOS
    (DETR-style ViT) model. Well-documented, but a different architecture
    from every other detector in this project (normalized box regression
    + a "no object" class + HF's own post-processing conventions), which
    would mean a bespoke decode path instead of the standard letterbox +
    NMS pipeline reused everywhere else.
  - spacewalk01/yolov5-face-mask-detection -- a custom-trained YOLOv5s,
    ships a ready mask_yolov5.pt directly in the GitHub repo. Chosen
    instead: it's the same Ultralytics-family single-stage detector
    convention as every sibling use case, so the existing letterbox/
    decode/NMS code applies unchanged.

The checkpoint predates the unified `ultralytics` pip package (it was
trained against the original standalone ultralytics/yolov5 repo) and
fails to unpickle through `YOLO(path)` (`ModuleNotFoundError: No module
named 'models'` -- the pickle references that repo's own models.yolo.Model
class). Exported instead with the classic yolov5 repo's own export.py
(see model/download.sh) -> model/mask_yolov5.onnx.

Detector I/O, confirmed empirically (fed both sample photos and inspected
raw output ranges before trusting the decode): input images[1,3,640,640]
RGB letterboxed, /255, standard Ultralytics-style preprocessing. Output
output0[1,25200,8] -- the classic YOLOv5 head convention (distinct from
YOLOv8/YOLO11's box+class-only head used elsewhere in this project): 4
box (cx,cy,w,h, already decoded to 640-space pixels via the anchor grid)
+ 1 objectness + 3 class scores, both objectness and class columns
confirmed bounded in [0,1] (sigmoid already applied in-graph). Final
per-box confidence is objectness * class_score (the standard YOLOv5
combination), not class_score alone.

Verified on two real photos: samples/sample_mask.jpg (Pexels/shvetsa, a
front-facing street portrait wearing a surgical mask correctly) scores
with_mask 0.96; samples/sample_no_mask.jpg (Pexels/olly, a comparable
front-facing portrait, bare face) scores without_mask 0.47 -- correct
class both times, though the second is a lower-confidence call, in line
with the author's own reported metrics (overall mAP@0.5=0.76; the
without_mask/with_mask classes are the strong ones, mask_weared_incorrect
is the weak one at mAP@0.5=0.43 due to that class's small share of the
853-image training set -- treat low-confidence "incorrect" calls with
extra skepticism).
"""

import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

CLASS_NAMES = ["with_mask", "without_mask", "mask_weared_incorrect"]
CLASS_COLORS = {
    "with_mask": (0, 200, 0),
    "without_mask": (0, 0, 255),
    "mask_weared_incorrect": (0, 165, 255),
}

# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIRS = ("model", "models", "weights", "onnx", "checkpoints", ".")
DEFAULT_HF_REPO = None  # no HF repo ships this checkpoint; exported model/mask_yolov5.onnx is committed locally

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
class FaceMaskDetector:
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
        preds = self.sess.run(self.output_names, {self.input_name: blob})[0][0]  # (25200, 8)
        boxes_xywh = preds[:, :4]
        objectness = preds[:, 4]
        class_scores = preds[:, 5:5 + len(CLASS_NAMES)]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = objectness * class_scores[np.arange(len(class_scores)), class_ids]
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
def draw_results(image, detections):
    out = image.copy()
    h, w = out.shape[:2]
    scale = max(w, h) / 1400
    box_thick = max(2, int(4 * scale))
    label_scale = max(0.9, 1.3 * scale)
    label_thick = max(2, int(3 * scale))

    counts = {name: 0 for name in CLASS_NAMES}
    for det in detections:
        counts[det["class"]] += 1
        color = CLASS_COLORS[det["class"]]
        x1, y1, x2, y2 = map(int, det["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thick)
        label = f'{det["class"]}  {det["score"]:.2f}'
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thick)
        label_y = max(th + baseline + 10, y1)
        cv2.rectangle(out, (x1, label_y - th - baseline - 10), (x1 + tw + 14, label_y), color, -1)
        cv2.putText(out, label, (x1 + 7, label_y - baseline - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, label_scale, (0, 0, 0), label_thick, cv2.LINE_AA)

    non_compliant = counts["without_mask"] + counts["mask_weared_incorrect"]
    banner = (f'FACES: {len(detections)}   WITH MASK: {counts["with_mask"]}   '
              f'WITHOUT: {counts["without_mask"]}   INCORRECT: {counts["mask_weared_incorrect"]}')
    banner_scale = max(1.0, 1.4 * scale)
    banner_thick = max(2, int(3 * scale))
    (tw, th), baseline = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, banner_scale, banner_thick)
    pad = int(14 * scale) + 8
    cv2.rectangle(out, (0, 0), (min(w, tw + pad * 2), th + baseline + pad * 2), (0, 0, 0), -1)
    color = (0, 0, 255) if non_compliant else (0, 255, 0)
    cv2.putText(out, banner, (pad, th + pad), cv2.FONT_HERSHEY_SIMPLEX, banner_scale, color, banner_thick, cv2.LINE_AA)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face mask compliance sanity test (single image).")
    parser.add_argument("-i", "--image", required=True)
    parser.add_argument("-o", "--output", default="output.jpg")
    parser.add_argument("-m", "--model", default=None,
                        help="Path to .onnx, a folder containing one, or a HF ref. "
                             "Omit to auto-discover in: " + ", ".join(DEFAULT_MODEL_DIRS))
    parser.add_argument("-c", "--conf-threshold", type=float, default=0.35)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    args = parser.parse_args()

    detector = FaceMaskDetector(resolve_model_path(args.model))

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)

    detections = detector.detect(image, confidence_threshold=args.conf_threshold,
                                  nms_threshold=args.nms_threshold)
    print(f"[detect] {len(detections)} face(s)")
    for det in detections:
        print(f'  {det["class"]}  conf={det["score"]:.2f}  box={[round(v) for v in det["box"]]}')

    annotated = draw_results(image, detections)
    cv2.imwrite(args.output, annotated)
    print(f"[done  ] wrote {args.output}")
