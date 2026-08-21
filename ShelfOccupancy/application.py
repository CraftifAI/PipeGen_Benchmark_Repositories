"""Single-image sanity test for retail shelf occupancy estimation.

No GitHub repo or existing model file for this use case, so Hugging Face
was searched: foduucom/product-detection-in-shelf-yolov8 (mAP@0.5 0.91),
classes {0: 'empty', 1: 'product'}. Its own ONNX export metadata shows it
was trained on a "Retail Coolers" dataset (refrigerated-drinks shelves), so
-- unlike the InventoryMonitoring use case, which needed this same model's
domain mismatch worked around -- this task's sample image is deliberately a
cooler/fridge shelf photo to match what the model actually saw in training.
The repo only ships best.pt, so it was exported once with
`YOLO("best.pt").export(format="onnx", opset=12, simplify=True)` (a
mechanical export, not training) -> model/best.onnx.

Unlike Inventory Monitoring (binary out-of-stock alert), Shelf Occupancy
reports a continuous fill percentage: the area covered by "product"
detections divided by the total area covered by "product" + "empty"
detections. This only measures occupancy within the parts of the shelf the
model actually recognized as one class or the other -- it is not a
segmentation of the whole shelf surface.

Ultralytics YOLOv8 detector, 2 classes: empty, product.
Input images[1,3,640,640] (RGB, letterbox to 640, grey (114,114,114) pad,
/255, no mean/std). Output output0[1,6,8400] = 4 bbox (cx,cy,w,h in
640-space) + 2 class scores, raw (nms:False) -> transpose + confidence
filter + NMS. Recipe from the embedded ONNX metadata: names
{0:'empty',1:'product'}, imgsz [640,640], task detect, nms False.
"""

import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

CLASS_NAMES = ["empty", "product"]


def load_image_exif_aware(path):
    """cv2.imread ignores EXIF orientation, so a phone photo shot in
    portrait but stored "sideways" (very common -- confirmed on one of this
    use case's own test images, EXIF orientation tag 6) gets processed
    rotated 90 degrees from how it displays, which tanks detection since
    the detector never saw sideways bottles in training. PIL + exif_transpose
    applies the rotation the camera recorded before handing off to cv2."""
    pil_img = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ---------------------------------------------------------------------------
# Step 2 — model path resolver (ships verbatim)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIRS = ("model", "models", "weights", "onnx", "checkpoints", ".")
# foduucom/product-detection-in-shelf-yolov8 ships only best.pt (no .onnx),
# so auto-fetch can't resolve it directly -- the exported model/best.onnx is
# committed locally instead.
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
        img = load_image_exif_aware(image_path)
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
        # output0: [1, 6, 8400] -> [8400, 6]; cols 0:4 box (cx,cy,w,h), 4:6 class scores
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
                    "class": CLASS_NAMES[class_ids[oi]],
                })
        return dets


def compute_occupancy(image_shape, detections):
    """Occupancy = product-covered area / (product + empty covered area),
    using pixel masks so overlapping boxes aren't double-counted."""
    h, w = image_shape[:2]
    product_mask = np.zeros((h, w), dtype=np.uint8)
    empty_mask = np.zeros((h, w), dtype=np.uint8)
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        if det["class"] == "product":
            product_mask[y1:y2, x1:x2] = 1
        else:
            empty_mask[y1:y2, x1:x2] = 1

    product_area = int(product_mask.sum())
    empty_area = int(empty_mask.sum())
    total = product_area + empty_area
    occupancy_pct = 100.0 * product_area / total if total > 0 else 0.0
    return occupancy_pct, product_area, empty_area


def draw_results(image_path, detections, occupancy_pct, output_path):
    image = load_image_exif_aware(image_path)
    h, w = image.shape[:2]
    scale = max(w, h) / 1400
    box_thick = max(2, int(4 * scale))
    label_scale = max(1.0, 1.3 * scale)
    label_thick = max(2, int(3 * scale))

    colors = {"product": (0, 200, 0), "empty": (0, 0, 255)}
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        color = colors[det["class"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thick)

    n_product = sum(1 for d in detections if d["class"] == "product")
    n_empty = sum(1 for d in detections if d["class"] == "empty")
    banner = f"SHELF OCCUPANCY: {occupancy_pct:.0f}%  (products:{n_product}  empty:{n_empty})"
    banner_scale = max(1.6, 2.1 * scale)
    banner_thick = max(3, int(4 * scale))
    pad = int(16 * scale) + 10
    # Shrink the banner until the whole string fits inside the frame: at
    # 2.1*scale a 2592px-wide photo needs ~3160px of text, so the product /
    # empty counts were being clipped off the right edge and never reached
    # the viewer. No-op when the text already fits.
    (tw, th), baseline = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, banner_scale, banner_thick)
    if tw + pad * 2 > w:
        banner_scale *= (w - pad * 2) / float(tw)
        banner_thick = max(2, int(round(banner_thick * (w - pad * 2) / float(tw))))
        (tw, th), baseline = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX,
                                             banner_scale, banner_thick)
    cv2.rectangle(image, (0, 0), (min(w, tw + pad * 2), th + baseline + pad * 2), (0, 0, 0), -1)
    if occupancy_pct >= 70:
        color = (0, 255, 0)
    elif occupancy_pct >= 35:
        color = (0, 200, 255)
    else:
        color = (0, 0, 255)
    cv2.putText(image, banner, (pad, th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, banner_scale, color, banner_thick, cv2.LINE_AA)
    # cv2.imwrite returns False instead of raising when the destination folder
    # does not exist, so the script used to print "[done] wrote ..." having
    # written nothing at all.
    parent = Path(output_path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(output_path, image):
        raise IOError(f"cv2.imwrite failed to write {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-image sanity test for shelf occupancy estimation.")
    parser.add_argument("-m", "--model", default=None,
                        help="Path to .onnx, a folder containing one, or a HF ref. "
                             "Omit to auto-discover in: " + ", ".join(DEFAULT_MODEL_DIRS))
    parser.add_argument("-i", "--image", required=True)
    parser.add_argument("-o", "--output", default="output.jpg")
    parser.add_argument("-c", "--conf-threshold", type=float, default=0.3)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    args = parser.parse_args()

    model = ModelTester(resolve_model_path(args.model))
    print(f"[input ] {model.input_name} {model.input_shape}")
    print(f"[output] {model.output_names}")

    outputs = model.run_inference(args.image)
    print(f"[raw   ] {outputs[0].shape}  min={outputs[0].min():.4f} max={outputs[0].max():.4f}")
    dets = model.postprocess(outputs, args.conf_threshold, args.nms_threshold)
    print(f"[detect] {len(dets)} detections: "
          f"{sum(1 for d in dets if d['class']=='product')} product, "
          f"{sum(1 for d in dets if d['class']=='empty')} empty")

    image = load_image_exif_aware(args.image)
    occupancy_pct, product_area, empty_area = compute_occupancy(image.shape, dets)
    print(f"[result] occupancy: {occupancy_pct:.1f}%  "
          f"(product_area={product_area}px  empty_area={empty_area}px)")

    draw_results(args.image, dets, occupancy_pct, args.output)
    print(f"[done  ] wrote {args.output}")
