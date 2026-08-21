"""Single-image sanity test for Enos-123/smoking-detection.

No GitHub repo exists for this use case. Searched GitHub first (mostly
research papers / no downloadable weights, or PPE-overlapping detectors
like yihong1120/Construction-Hazard-Detection), then Hugging Face.

First tried MeetCool/yolov8n-smoking-detection (classes cigarette/person/
smoke, ready-made ONNX) but it scored the cigarette class near-zero
(<0.01) on three different real photos with a clearly visible cigarette,
while confidently detecting "person" through the same preprocessing —
ruling out a decode bug and pointing to a weak/overfit fine-tune. Enos-123/
smoking-detection (single class: cigarette), trained on the same
"Smoking-Person-Detection" Roboflow dataset family, correctly detected the
cigarette in all three (0.48, 0.76, 0.28 confidence) using Ultralytics
directly. That repo only ships best.pt, so it was exported once with
`YOLO(weights).export(format="onnx", opset=12, simplify=True)` (a
mechanical export, not training) -> model/best.onnx.

Ultralytics YOLO11m detector, 1 class: cigarette.
Input images[1,3,640,640] (RGB, letterbox to 640 with grey (114,114,114)
padding, /255, no mean/std). Output output0[1,5,8400] = 4 bbox (cx,cy,w,h in
640-space) + 1 class score, raw (export used nms:False) -> needs transpose +
confidence filter + NMS.
Recipe/contract taken entirely from the embedded ONNX metadata (custom_metadata_map):
  names: {0: 'cigarette'}, imgsz: [640, 640], task: detect, nms: False --
  standard Ultralytics export convention.
"""

import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# names from embedded ONNX metadata (custom_metadata_map['names'])
CLASS_NAMES = ["cigarette"]
UNSAFE_CLASSES = {"cigarette"}  # single-class model — every detection is the unsafe behavior

# ---------------------------------------------------------------------------
# Step 2 — model path resolver (ships verbatim)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIRS = ("model", "models", "weights", "onnx", "checkpoints", ".")
# Enos-123/smoking-detection ships only best.pt (no .onnx), so auto-fetch can't
# resolve it directly — the exported model/best.onnx is committed locally instead.
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
        """YOLOv8 letterbox: RGB, aspect-preserving resize to 640, grey pad, /255."""
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
        colors = {"cigarette": (0, 0, 255)}
        unsafe_count = 0
        for det in detections:
            x1, y1, x2, y2 = map(int, det["box"])
            name = CLASS_NAMES[det["class"]]
            if name in UNSAFE_CLASSES:
                unsafe_count += 1
            color = colors.get(name, (255, 255, 255))
            label = f'{name} {det["score"]:.2f}'
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        banner = f"UNSAFE BEHAVIOR (smoking): {unsafe_count}" if unsafe_count else "no unsafe behavior detected"
        cv2.rectangle(image, (0, 0), (520, 40), (0, 0, 0), -1)
        cv2.putText(image, banner, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if unsafe_count else (0, 255, 0), 2)
        cv2.imwrite(output_path, image)
        return unsafe_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-image sanity test for the smoking-detection (cigarette) ONNX model.")
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
        print(f"         {CLASS_NAMES[d['class']]:<10} {d['score']:.3f}  "
              f"[{d['box'][0]:.0f},{d['box'][1]:.0f},{d['box'][2]:.0f},{d['box'][3]:.0f}]")
    unsafe_count = model.draw_detections(args.image, dets, args.output)
    print(f"[result] unsafe behavior flagged: {unsafe_count}")
    print(f"[done  ] wrote {args.output}")
