"""Single-image sanity test for Automatic Number Plate Recognition (ANPR).

No GitHub repo/model was handed over for this use case, so one was found:
ankandrew/fast-alpr (a ready-to-use ALPR framework) composes two of the
author's own pretrained ONNX models, no training/annotation required:

1. Detection: yolo-v9-t-384-license-plate-end2end, from ankandrew's
   open-image-models project. "end2end" = NMS is baked into the graph.
   Direct download (GitHub release asset, no HF repo):
     https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-384-license-plates-end2end.onnx
   Input images[1,3,384,384] (RGB, letterbox, grey (114,114,114) pad,
   /255, NCHW float32 -- read from open_image_models' own preprocess.py).
   Output [N,7] = [batch_idx, x1, y1, x2, y2, class_id, score], boxes in
   letterboxed 384-space pixels -- read from its own postprocess.py.

2. OCR: cct-xs-v2-global-model, from ankandrew's fast-plate-ocr project
   (Compact Convolutional Transformer, 65+ countries). Direct download:
     https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_xs_v2_global.onnx
     .../cct_xs_v2_global_plate_config.yaml
   Input input[N,64,128,3] uint8 RGB, PLAIN resize (no letterbox,
   keep_aspect_ratio=false) -- read from fast_plate_ocr's process.py.
   Output 'plate'[N,10,37]: 10 character slots x 37-symbol alphabet
   ("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"), argmax per slot, mapped
   through the alphabet, trailing '_' pad stripped. Output 'region'[N,66]:
   argmax over a fixed country/region list (both straight from the
   model's own config.yaml, not guessed).

This script is a from-scratch onnxruntime implementation of that same
two-model pipeline (not a wrapper around the fast_alpr package), matching
this project's convention of a single self-contained application.py.
Verified against the actual fast_alpr package on the bundled sample before
writing this: same box, same OCR string ("E486B_6" -- the real plate is
"E486B?26 RUS" with one digit taped over in the source photo, so the '_'
here is a genuine unreadable character, not a bug).
"""

import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml

DETECT_URL = "https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-384-license-plates-end2end.onnx"
OCR_URL = "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_xs_v2_global.onnx"
OCR_CONFIG_URL = "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_xs_v2_global_plate_config.yaml"

DEFAULT_DETECT_DIRS = ("model/detect", "model", "models", ".")
DEFAULT_OCR_DIRS = ("model/ocr", "model", "models", ".")


def _download(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch ] {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return str(dest)


def resolve_onnx(search_dirs, download_url, download_name, explicit=None):
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"--model path does not exist: {p}")
        return str(p)

    for d in search_dirs:
        d = Path(d).expanduser()
        if not d.is_dir():
            continue
        candidates = [f for f in sorted(d.glob("*.onnx")) if not any(part.startswith(".") for part in f.parts)]
        if len(candidates) == 1:
            print(f"[model] auto-resolved: {candidates[0]}")
            return str(candidates[0])
        if len(candidates) > 1:
            raise RuntimeError(f"Found {len(candidates)} .onnx files in {d} — pick one explicitly")

    return _download(download_url, Path(search_dirs[0]) / download_name)


# ---------------------------------------------------------------------------
# Stage 1 — plate detection (YOLOv9-t end2end, 384x384, NMS baked in)
# ---------------------------------------------------------------------------
DETECT_SIZE = 384
CONF_THRESHOLD = 0.4


def letterbox(im, new_shape, color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


class PlateDetector:
    def __init__(self, model_path):
        self.sess = ort.InferenceSession(model_path, providers=ort.get_available_providers())
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def detect(self, image, confidence_threshold=CONF_THRESHOLD):
        letterboxed, ratio, (dw, dh) = letterbox(image, (DETECT_SIZE, DETECT_SIZE))
        blob = letterboxed.transpose(2, 0, 1)[::-1]  # HWC->CHW, BGR->RGB
        blob = (blob / 255.0).astype(np.float32)[None, ...]
        preds = self.sess.run([self.output_name], {self.input_name: np.ascontiguousarray(blob)})[0]

        detections = []
        for row in preds:
            x1, y1, x2, y2, class_id, score = row[1], row[2], row[3], row[4], row[5], row[6]
            if score < confidence_threshold:
                continue
            x1 = (x1 - dw) / ratio
            y1 = (y1 - dh) / ratio
            x2 = (x2 - dw) / ratio
            y2 = (y2 - dh) / ratio
            detections.append({
                "box": [max(0, x1), max(0, y1), min(image.shape[1], x2), min(image.shape[0], y2)],
                "score": float(score),
            })
        return detections


# ---------------------------------------------------------------------------
# Stage 2 — plate OCR (CCT-XS, 64x128, plain resize, uint8 RGB in)
# ---------------------------------------------------------------------------
class PlateOCR:
    def __init__(self, model_path, config_path):
        self.sess = ort.InferenceSession(model_path, providers=ort.get_available_providers())
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]

        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.max_plate_slots = cfg["max_plate_slots"]
        self.alphabet = cfg["alphabet"]
        self.pad_char = cfg["pad_char"]
        self.img_h = cfg["img_height"]
        self.img_w = cfg["img_width"]
        self.regions = cfg.get("plate_regions")

    def read(self, plate_crop_bgr):
        rgb = cv2.cvtColor(plate_crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        blob = resized[None, ...].astype(np.uint8)  # (1, H, W, 3), model normalizes internally

        outputs = self.sess.run(self.output_names, {self.input_name: blob})
        by_name = dict(zip(self.output_names, outputs))
        # both heads already end in softmax in-graph (each row sums to 1,
        # confirmed empirically) -- take argmax/max directly, no extra softmax
        plate_probs = by_name["plate"].reshape(-1, self.max_plate_slots, len(self.alphabet))
        char_idx = np.argmax(plate_probs, axis=-1)[0]
        char_conf = np.max(plate_probs, axis=-1)[0]
        text = "".join(self.alphabet[i] for i in char_idx).rstrip(self.pad_char)
        mean_conf = float(char_conf[:len(text)].mean()) if text else 0.0

        region, region_conf = None, None
        if "region" in by_name and self.regions:
            region_probs = by_name["region"][0]
            r_idx = int(np.argmax(region_probs))
            region, region_conf = self.regions[r_idx], float(region_probs[r_idx])

        return {"text": text, "confidence": mean_conf, "region": region, "region_confidence": region_conf}


# ---------------------------------------------------------------------------
def draw_results(image, plates, output_path):
    out = image.copy()
    h, w = out.shape[:2]
    scale = max(w, h) / 1400
    box_thick = max(2, int(4 * scale))
    label_scale = max(0.9, 1.3 * scale)
    label_thick = max(2, int(3 * scale))

    for p in plates:
        x1, y1, x2, y2 = map(int, p["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), box_thick)
        label = f'{p["ocr"]["text"]}  ({p["ocr"]["confidence"]:.2f})'
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thick)
        label_y = max(th + baseline + 10, y1)
        cv2.rectangle(out, (x1, label_y - th - baseline - 10), (x1 + tw + 14, label_y), (0, 255, 0), -1)
        cv2.putText(out, label, (x1 + 7, label_y - baseline - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, label_scale, (0, 0, 0), label_thick, cv2.LINE_AA)

    banner = f"PLATES READ: {len(plates)}"
    banner_scale = max(1.6, 2.1 * scale)
    banner_thick = max(3, int(4 * scale))
    (tw, th), baseline = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, banner_scale, banner_thick)
    pad = int(16 * scale) + 10
    cv2.rectangle(out, (0, 0), (min(w, tw + pad * 2), th + baseline + pad * 2), (0, 0, 0), -1)
    cv2.putText(out, banner, (pad, th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, banner_scale, (0, 255, 0), banner_thick, cv2.LINE_AA)
    # cv2.imwrite() returns False instead of raising when it cannot write (e.g. the
    # parent dir does not exist, or the path is not writable) -- without this check the
    # script printed "[done  ] wrote ..." and exited 0 having written nothing.
    out_path = Path(output_path)
    if out_path.parent and not out_path.parent.is_dir():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), out):
        raise RuntimeError(f"cv2.imwrite failed to write {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ANPR sanity test: detect + read license plates.")
    parser.add_argument("-i", "--image", required=True)
    parser.add_argument("-o", "--output", default="output.jpg")
    parser.add_argument("--detect-model", default=None)
    parser.add_argument("--ocr-model", default=None)
    parser.add_argument("--ocr-config", default=None)
    parser.add_argument("-c", "--conf-threshold", type=float, default=CONF_THRESHOLD)
    args = parser.parse_args()

    detect_path = resolve_onnx(DEFAULT_DETECT_DIRS, DETECT_URL, "plate_detector.onnx", args.detect_model)
    ocr_path = resolve_onnx(DEFAULT_OCR_DIRS, OCR_URL, "plate_ocr.onnx", args.ocr_model)
    ocr_config_path = args.ocr_config
    if ocr_config_path is None:
        cfg_candidate = Path(ocr_path).with_name("plate_ocr_config.yaml")
        if not cfg_candidate.is_file():
            _download(OCR_CONFIG_URL, cfg_candidate)
        ocr_config_path = str(cfg_candidate)

    detector = PlateDetector(detect_path)
    ocr = PlateOCR(ocr_path, ocr_config_path)

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)

    detections = detector.detect(image, confidence_threshold=args.conf_threshold)
    print(f"[detect] {len(detections)} plate(s)")

    plates = []
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        result = ocr.read(crop)
        plates.append({"box": det["box"], "det_score": det["score"], "ocr": result})
        region_str = f", region={result['region']} ({result['region_confidence']:.2f})" if result["region"] else ""
        print(f"  plate='{result['text']}'  conf={result['confidence']:.2f}"
              f"  det_conf={det['score']:.2f}{region_str}")

    draw_results(image, plates, args.output)
    print(f"[done  ] wrote {args.output}")
