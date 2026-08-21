"""Single-image (or multi-frame) sanity test for checkout-lane monitoring.

No dedicated "checkout monitoring" model exists (searched Hugging Face for
"checkout monitoring", "self-checkout", "POS monitoring", "cashier
detection", "retail checkout" -- only unrelated Stable-Diffusion LoRA
repos under "openrail" license and one empty stub repo turned up). Same
situation as QueueLengthMonitoring/DockUtilization: this is a
zone-membership rule on top of person detection, not a trained visual
class. Reuses the already-verified Ultralytics YOLO11s (COCO-pretrained)
person detector from VehicleDetection/DockUtilization/
QueueLengthMonitoring/CustomerDwellTime.

What this adds over QueueLengthMonitoring's single generic zone: a
checkout lane has two functionally different areas -- the cashier's spot
and the customer queue -- and the actionable question is usually "which
lanes are actually open/staffed right now", not just a raw headcount.
So each lane in --zone-file gets its OWN pair of polygons:
  - cashier_zone: a person here means the lane is staffed.
  - queue_zone: people here are customers waiting at that lane.
A lane's status is then one of:
  OPEN - SERVING     cashier present, 1+ customers queued
  OPEN - IDLE         cashier present, nobody queued
  CLOSED               no cashier, nobody queued
  UNSTAFFED - QUEUE!  no cashier BUT customers are queued anyway (the one
                       state worth a manager's attention -- a lane that
                       should be opened)
Omit --zone-file for a single generic whole-frame lane with no
cashier_zone (staffing unknown), matching QueueLengthMonitoring's
no-zone-given fallback.

Verified on a real wide-angle photo of a multi-lane checkout row
(samples/sample.jpg): people this far from a ceiling-mounted camera are
small, so confidence is genuinely lower here than in closer-range use
cases (0.10-0.61 across the 3 real people in that photo) -- pass a lower
--conf-threshold for this kind of wide/overhead camera placement (the
sample was run at -c 0.15); this is a property of the shot, not a decode
bug (checked against QueueLengthMonitoring's identical decode contract).

Detector I/O: identical to the sibling use cases (input images[1,3,640,640]
RGB letterboxed, /255; output [1,84,8400] raw 4-box+80-class COCO, needs
transpose + confidence filter + NMS).
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
DEFAULT_HF_REPO = None  # Ultralytics/YOLO11 ships only .pt; exported model/yolo11s.onnx is committed locally

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

    def detect(self, img, confidence_threshold=0.25, nms_threshold=0.45):
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
def foot_point(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, y2)  # bottom-center of the box approximates where the person stands


def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.float32), point, False) >= 0


def default_lanes(img_shape):
    """A single generic whole-frame lane, no cashier_zone -- used when no
    --zone-file is given. Staffing can't be determined without a cashier
    zone, so status only ever reports the queue count."""
    h, w = img_shape[:2]
    return [{"name": "Lane 1", "cashier_zone": None, "queue_zone": [[0, 0], [w, 0], [w, h], [0, h]]}]


def lane_status(cashier_present, queue_count, has_cashier_zone):
    if not has_cashier_zone:
        return "QUEUE" if queue_count else "EMPTY"
    if cashier_present and queue_count:
        return "OPEN - SERVING"
    if cashier_present:
        return "OPEN - IDLE"
    if queue_count:
        return "UNSTAFFED - QUEUE!"
    return "CLOSED"


STATUS_COLOR = {
    "OPEN - SERVING": (0, 200, 0),
    "OPEN - IDLE": (0, 200, 0),
    "CLOSED": (150, 150, 150),
    "UNSTAFFED - QUEUE!": (0, 0, 255),
    "QUEUE": (0, 140, 255),
    "EMPTY": (150, 150, 150),
}


def evaluate_lanes(lanes, people):
    results = []
    for i, lane in enumerate(lanes, start=1):
        # .get(): an omitted key means "no such zone", same as an explicit null,
        # instead of an opaque KeyError on an otherwise usable --zone-file.
        cashier_zone, queue_zone = lane.get("cashier_zone"), lane.get("queue_zone")
        cashier_here, queue_here = [], []
        for p in people:
            fp = foot_point(p["box"])
            if cashier_zone and point_in_polygon(fp, cashier_zone):
                cashier_here.append(p)
            elif queue_zone and point_in_polygon(fp, queue_zone):
                queue_here.append(p)
        status = lane_status(bool(cashier_here), len(queue_here), cashier_zone is not None)
        results.append({
            "name": lane.get("name", f"Lane {i}"), "status": status,
            "cashier_count": len(cashier_here), "queue_count": len(queue_here),
            "cashier_people": cashier_here, "queue_people": queue_here,
        })
    return results


# ---------------------------------------------------------------------------
def draw_results(image, lane_results, lanes, output_path):
    out = image.copy()
    h, w = out.shape[:2]
    # Text/line sizing is driven by WIDTH, not max(w,h): on a tall portrait
    # frame max(w,h) massively overscales banner text past the frame edge.
    scale = w / 1400
    box_thick = max(2, int(3 * scale))
    marker_scale = max(0.9, 1.3 * scale)
    marker_thick = max(2, int(3 * scale))

    for idx, (lane, res) in enumerate(zip(lanes, lane_results), start=1):
        color = STATUS_COLOR[res["status"]]
        if lane.get("cashier_zone"):
            cv2.polylines(out, [np.array(lane["cashier_zone"], dtype=np.int32)], True, (255, 200, 0), max(2, int(3 * scale)))
        if lane.get("queue_zone"):
            pts = np.array(lane["queue_zone"], dtype=np.int32)
            cv2.polylines(out, [pts], True, (255, 0, 255), max(2, int(3 * scale)))
            # A single numbered marker at the zone centroid -- cross-referenced
            # against the legend -- instead of a text label per zone, since
            # checkout lanes sit close together and full labels would overlap.
            cx, cy = pts.mean(axis=0).astype(int)
            marker = str(idx)
            (tw, th), baseline = cv2.getTextSize(marker, cv2.FONT_HERSHEY_SIMPLEX, marker_scale, marker_thick)
            radius = max(tw, th) // 2 + int(10 * scale)
            cv2.circle(out, (cx, cy), radius, color, -1)
            cv2.circle(out, (cx, cy), radius, (255, 255, 255), max(1, int(2 * scale)))
            cv2.putText(out, marker, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        marker_scale, (0, 0, 0), marker_thick, cv2.LINE_AA)

        for p in res["cashier_people"]:
            x1, y1, x2, y2 = map(int, p["box"])
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), box_thick)
        for p in res["queue_people"]:
            x1, y1, x2, y2 = map(int, p["box"])
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 255), box_thick)

    open_count = sum(1 for r in lane_results if r["status"].startswith("OPEN"))
    unstaffed_alert = sum(1 for r in lane_results if r["status"] == "UNSTAFFED - QUEUE!")
    total_queue = sum(r["queue_count"] for r in lane_results)
    banner = f"LANES OPEN: {open_count}/{len(lane_results)}   WAITING: {total_queue}"
    if unstaffed_alert:
        banner += f"   UNSTAFFED W/ QUEUE: {unstaffed_alert}"
    banner_scale = max(0.7, 1.0 * scale)
    banner_thick = max(2, int(2 * scale))
    (tw, th), baseline = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, banner_scale, banner_thick)
    pad = int(10 * scale) + 6
    banner_color = (0, 0, 255) if unstaffed_alert else (0, 255, 0)
    cv2.rectangle(out, (0, 0), (min(w, tw + pad * 2), th + baseline + pad * 2), (0, 0, 0), -1)
    cv2.putText(out, banner, (pad, th + pad), cv2.FONT_HERSHEY_SIMPLEX, banner_scale, banner_color, banner_thick, cv2.LINE_AA)

    # Legend: numbered marker -> lane name + status, stacked below the banner.
    ly = th + baseline + pad * 2 + int(8 * scale)
    for idx, res in enumerate(lane_results, start=1):
        color = STATUS_COLOR[res["status"]]
        line = f'{idx}. {res["name"]}: {res["status"]}'
        (tw2, th2), baseline2 = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, banner_scale, banner_thick)
        cv2.rectangle(out, (0, ly), (min(w, tw2 + pad * 2), ly + th2 + baseline2 + int(6 * scale)), (0, 0, 0), -1)
        cv2.putText(out, line, (pad, ly + th2 + int(2 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    banner_scale, color, banner_thick, cv2.LINE_AA)
        ly += th2 + baseline2 + int(10 * scale)

    # cv2.imwrite() returns False (it does not raise) when the target directory
    # does not exist, which otherwise let the script print "[done ] wrote ..."
    # and exit 0 having written nothing at all.
    if not cv2.imwrite(output_path, out):
        raise IOError(f"cv2.imwrite failed to write {output_path} "
                      "(does the parent directory exist, and is the extension supported?)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Checkout-lane monitoring sanity test.")
    parser.add_argument("-m", "--model", default=None,
                        help="Path to .onnx, a folder containing one, or a HF ref. "
                             "Omit to auto-discover in: " + ", ".join(DEFAULT_MODEL_DIRS))
    parser.add_argument("-i", "--image", required=True)
    parser.add_argument("-o", "--output", default="output.jpg")
    parser.add_argument("-c", "--conf-threshold", type=float, default=0.25)
    parser.add_argument("--zone-file", default=None,
                         help='JSON: {"lanes": [{"name": str, "cashier_zone": [[x,y],...] or null, '
                              '"queue_zone": [[x,y],...]}]}. Omit for a single whole-frame lane '
                              "with no cashier zone (staffing unknown).")
    args = parser.parse_args()

    model = PersonDetector(resolve_model_path(args.model))
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)

    detections = model.detect(image, confidence_threshold=args.conf_threshold)
    people = [d for d in detections if d["class"] == PERSON_CLASS]
    print(f"[detect] {len(detections)} object(s), {len(people)} person(s)")

    if args.zone_file:
        lanes = json.load(open(args.zone_file))["lanes"]
        print(f"[zones ] {len(lanes)} lane(s) loaded from {args.zone_file}")
    else:
        lanes = default_lanes(image.shape)
        print("[zones ] no --zone-file given, using a single whole-frame lane")

    lane_results = evaluate_lanes(lanes, people)
    for res in lane_results:
        print(f"  {res['name']}: {res['status']}  (cashier={res['cashier_count']}, queue={res['queue_count']})")

    draw_results(image, lane_results, lanes, args.output)
    open_count = sum(1 for r in lane_results if r["status"].startswith("OPEN"))
    print(f"[result] {open_count}/{len(lane_results)} lane(s) open, "
          f"{sum(r['queue_count'] for r in lane_results)} customer(s) waiting")
    print(f"[done  ] wrote {args.output}")
