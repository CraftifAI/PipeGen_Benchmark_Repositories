"""Multi-frame sanity test for industrial restricted-area / danger-zone
intrusion monitoring (Surveillance > Security).

No dedicated "restricted area" or "intrusion detection" model exists on
Hugging Face -- searched "restricted area", "intrusion detection",
"perimeter intrusion", "trespassing detection", "unauthorized access
detection"; every real hit was a NETWORK/cybersecurity intrusion-detection
model (text classifiers over traffic logs), not visual CCTV. Same
situation as QueueLengthMonitoring/CheckoutMonitoring: this is a
zone-membership rule on top of person detection, not a trained visual
class. Reuses the already-verified Ultralytics YOLO11s (COCO-pretrained)
person detector from VehicleDetection/DockUtilization/CustomerDwellTime,
the SORT-style IOU+Hungarian tracker from ContainerTracking/
CustomerDwellTime/TheftDetection (persistent ID across frames), and the
polygon zone-membership check from QueueLengthMonitoring/CheckoutMonitoring
/CustomerDwellTime.

What's specific to this use case: unlike a retail queue/dwell zone
(harmless to be in), a restricted/danger zone means ANY presence is a
security-relevant event, not just prolonged presence. So every tracked
person's status is one of:
  CLEAR      outside every zone
  IN ZONE     inside a zone, but hasn't yet cleared --alert-frames
  INTRUSION!  inside a zone for >= --alert-frames consecutive frames --
              flagged red, with how many seconds they've been in there
              (reusing CustomerDwellTime's cumulative-seconds bookkeeping,
              since "how long has the intruder been in the danger zone"
              is exactly the number a security operator needs next).
--alert-frames defaults to 2 (not TheftDetection's 3) because zone
membership is deterministic geometry, not a noisy action classifier --
the debounce here only needs to absorb a stray missed detection at the
zone boundary, not classifier flicker.

Verified on a real factory-floor photo (samples/sample.jpg, two workers
near glass-cutting machinery): the worker with hands directly on the
running machine (foot-point deep inside the marked danger zone) is
correctly flagged INTRUSION!, while their coworker standing clear of the
machine (foot-point outside the zone) is correctly left CLEAR --
samples/output.jpg. Omit --zone-file for a whole-frame zone (every person
anywhere in view counts as an intrusion) -- appropriate for a camera
whose entire field of view is a restricted area (e.g. a locked server
room), rather than one where only part of the frame is hazardous.

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
from scipy.optimize import linear_sum_assignment

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
DEFAULT_FPS_FALLBACK = 25.0

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
# Minimal SORT-style tracker: IOU + Hungarian matching, no appearance model
# (same approach as ContainerTracking/CustomerDwellTime/TheftDetection).
# ---------------------------------------------------------------------------
def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def foot_point(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, y2)  # bottom-center of the box approximates where the person stands


def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.float32), point, False) >= 0


def in_any_zone(box, zones):
    fp = foot_point(box)
    for zone in zones:
        if point_in_polygon(fp, zone["polygon"]):
            return zone["name"]
    return None


def default_zones(img_shape):
    """Whole frame -- used when no --zone-file is given, i.e. the camera's
    entire field of view is treated as the restricted area."""
    h, w = img_shape[:2]
    return [{"name": "Restricted Area", "polygon": [[0, 0], [w, 0], [w, h], [0, h]]}]


class Track:
    def __init__(self, track_id, detection):
        self.id = track_id
        self.box = detection["box"]
        self.score = detection["score"]
        self.misses = 0
        self.hits = 1
        self.trail = [self._center()]

    def _center(self):
        x1, y1, x2, y2 = self.box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def update(self, detection):
        self.box = detection["box"]
        self.score = detection["score"]
        self.hits += 1
        self.misses = 0
        self.trail.append(self._center())
        self.trail = self.trail[-30:]


class PersonTracker:
    def __init__(self, iou_threshold=0.3, max_age=15):
        self.tracks = []
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    def update(self, detections):
        if not self.tracks:
            for det in detections:
                self.tracks.append(Track(self.next_id, det))
                self.next_id += 1
            return list(self.tracks)

        if detections:
            cost = np.zeros((len(self.tracks), len(detections)))
            for i, trk in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    cost[i, j] = 1 - iou(trk.box, det["box"])
            row_idx, col_idx = linear_sum_assignment(cost)
        else:
            row_idx, col_idx = np.array([], dtype=int), np.array([], dtype=int)

        matched_tracks, matched_dets = set(), set()
        for r, c in zip(row_idx, col_idx):
            if 1 - cost[r, c] >= self.iou_threshold:
                self.tracks[r].update(detections[c])
                matched_tracks.add(r)
                matched_dets.add(c)

        for i, trk in enumerate(self.tracks):
            if i not in matched_tracks:
                trk.misses += 1

        for j, det in enumerate(detections):
            if j not in matched_dets:
                self.tracks.append(Track(self.next_id, det))
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
        return list(self.tracks)


# ---------------------------------------------------------------------------
class IntrusionRecord:
    """Per-track-ID bookkeeping: consecutive-frame streak inside any zone
    (for alert debounce) plus cumulative seconds inside any zone (for
    "how long has this intrusion lasted", the number a security operator
    needs). Kept independent of the tracker's own track list so stats
    survive after a track is pruned."""

    def __init__(self):
        self.streak = 0
        self.frames_in_zone = 0
        self.zone_name = None
        self.last_zone_name = None
        self.peak_streak = 0

    def update(self, zone_name):
        if zone_name:
            self.frames_in_zone += 1
            self.streak += 1
            self.zone_name = zone_name
            self.last_zone_name = zone_name
        else:
            self.streak = 0
            self.zone_name = None
        self.peak_streak = max(self.peak_streak, self.streak)

    def is_intrusion(self, alert_frames):
        return self.streak >= alert_frames

    def ever_intrusion(self, alert_frames):
        """Did this track EVER hold an alert-length streak inside a zone?
        `is_intrusion` is the live per-frame state (resets the moment the
        person steps back out, which is what the overlay must show); the
        end-of-run summary instead has to report everyone who intruded at
        any point, otherwise an intruder who walks out before the last
        frame disappears from the report entirely."""
        return self.peak_streak >= alert_frames

    def dwell_seconds(self, fps):
        return self.frames_in_zone / fps


# ---------------------------------------------------------------------------
def draw_frame(image, zones, tracks, records, alert_frames, fps):
    out = image.copy()
    h, w = out.shape[:2]
    scale = w / 1400
    box_thick = max(2, int(3 * scale))
    label_scale = max(0.8, 1.1 * scale)
    label_thick = max(1, int(2 * scale))

    for zone in zones:
        cv2.polylines(out, [np.array(zone["polygon"], dtype=np.int32)], True, (0, 0, 255), max(2, int(4 * scale)))

    intruder_count, longest = 0, (None, 0.0)
    for trk in tracks:
        if trk.misses > 0:
            continue
        rec = records[trk.id]
        alert = rec.is_intrusion(alert_frames)
        intruder_count += int(alert)
        dwell_s = rec.dwell_seconds(fps)
        if alert and dwell_s > longest[1]:
            longest = (trk.id, dwell_s)

        if alert:
            color = (0, 0, 255)
        elif rec.zone_name:
            color = (0, 140, 255)
        else:
            color = (0, 200, 0)

        x1, y1, x2, y2 = map(int, trk.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thick)
        for i in range(1, len(trk.trail)):
            cv2.line(out, trk.trail[i - 1], trk.trail[i], color, max(1, box_thick // 2))

        label = f"ID {trk.id}  " + (f"INTRUSION! {dwell_s:.1f}s" if alert else ("IN ZONE" if rec.zone_name else "CLEAR"))
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thick)
        label_y = max(th + baseline + 8, y1)
        cv2.rectangle(out, (x1, label_y - th - baseline - 8), (x1 + tw + 12, label_y), color, -1)
        cv2.putText(out, label, (x1 + 6, label_y - baseline - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, label_scale, (0, 0, 0), label_thick, cv2.LINE_AA)

    banner = f"ZONES: {len(zones)}   INTRUSIONS: {intruder_count}"
    if longest[0] is not None:
        banner += f"   LONGEST: ID {longest[0]} {longest[1]:.1f}s"
    banner_scale = max(0.8, 1.1 * scale)
    banner_thick = max(2, int(3 * scale))
    (tw, th), baseline = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, banner_scale, banner_thick)
    pad = int(10 * scale) + 6
    cv2.rectangle(out, (0, 0), (min(w, tw + pad * 2), th + baseline + pad * 2), (0, 0, 0), -1)
    color = (0, 0, 255) if intruder_count else (0, 255, 0)
    cv2.putText(out, banner, (pad, th + pad), cv2.FONT_HERSHEY_SIMPLEX, banner_scale, color, banner_thick, cv2.LINE_AA)
    return out


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_frames(source):
    p = Path(source)
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(source)
        return [(p.stem, img)], None

    if p.is_dir():
        paths = sorted(p.glob("*.jpg")) + sorted(p.glob("*.png"))
        return [(str(fp), cv2.imread(str(fp))) for fp in paths], None

    cap = cv2.VideoCapture(str(p))
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append((f"frame_{idx:04d}", frame))
        idx += 1
    cap.release()
    return frames, fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Restricted-area / danger-zone intrusion monitoring sanity test (image, video, or frame directory in).")
    parser.add_argument("-i", "--input", required=True,
                        help="Path to a single image, a video file, OR a directory of ordered "
                             "frame images.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path: an image for single-image input, a video for video/"
                             "frame-dir input (or a directory if --frames is set). "
                             "Default: output.jpg / intrusion_tracked.mp4")
    parser.add_argument("-m", "--model", default=None,
                        help="Path to .onnx, a folder containing one, or a HF ref. "
                             "Omit to auto-discover in: " + ", ".join(DEFAULT_MODEL_DIRS))
    parser.add_argument("-c", "--conf-threshold", type=float, default=0.35)
    parser.add_argument("--zone-file", default=None,
                         help='JSON: {"zones": [{"name": str, "polygon": [[x,y],...]}]}. '
                              "Omit for a whole-frame zone.")
    parser.add_argument("--alert-frames", type=int, default=2,
                         help="Consecutive frames a person must be inside a zone before an "
                              "INTRUSION alert is raised for their track (default: 2)")
    parser.add_argument("--iou-threshold", type=float, default=0.3,
                         help="Minimum IOU for the tracker to match a detection to an existing track")
    parser.add_argument("--max-age", type=int, default=15,
                         help="Frames a track survives without a matching detection")
    parser.add_argument("--fps", type=float, default=None,
                         help="Override the video's reported fps (used to convert frames to "
                              "seconds); needed when a source reports 0/unreliable fps")
    parser.add_argument("--frames", action="store_true",
                         help="Write annotated frames to --output (a directory) instead of a video")
    args = parser.parse_args()

    single_image = Path(args.input).is_file() and Path(args.input).suffix.lower() in IMAGE_EXTS
    if single_image and args.frames:
        raise ValueError("--frames only applies to video/frame-directory input")
    if args.output is None:
        args.output = "output.jpg" if single_image else "intrusion_tracked.mp4"
    alert_frames = 1 if single_image else args.alert_frames

    detector = PersonDetector(resolve_model_path(args.model))
    tracker = PersonTracker(iou_threshold=args.iou_threshold, max_age=args.max_age)

    frames, detected_fps = load_frames(args.input)
    if not frames:
        raise FileNotFoundError(f"No frames found at {args.input}")
    fps = args.fps or detected_fps or DEFAULT_FPS_FALLBACK
    if not (args.fps or detected_fps):
        print(f"[fps   ] source reported no usable fps, falling back to {fps:.1f} -- "
              f"pass --fps for accurate dwell times")
    print(f"[input ] {len(frames)} frame(s) from {args.input} @ {fps:.1f} fps")
    if single_image:
        print("[note  ] single-image input -- showing the raw per-frame zone membership, no "
              "multi-frame alert debounce (run on a video for that)")

    if args.zone_file:
        zones = json.load(open(args.zone_file))["zones"]
        print(f"[zones ] {len(zones)} zone(s) loaded from {args.zone_file}")
    else:
        zones = default_zones(frames[0][1].shape)
        print("[zones ] no --zone-file given, treating the whole frame as restricted")

    if single_image:
        pass  # single annotated frame is written directly after the loop, no writer needed
    elif args.frames:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        h, w = frames[0][1].shape[:2]
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), min(fps, 30), (w, h))

    records = {}
    last_annotated = None
    for frame_idx, (name, frame) in enumerate(frames):
        detections = detector.detect(frame, confidence_threshold=args.conf_threshold)
        people = [d for d in detections if d["class"] == PERSON_CLASS]
        tracks = tracker.update(people)

        active = [t for t in tracks if t.misses == 0]
        for trk in active:
            if trk.id not in records:
                records[trk.id] = IntrusionRecord()
            zone_name = in_any_zone(trk.box, zones)
            records[trk.id].update(zone_name)

        alerts = [t.id for t in active if records[t.id].is_intrusion(alert_frames)]
        print(f"  {name}: {len(active)} person(s) -> "
              f"{[f'ID{t.id}' for t in active]}"
              + (f"  INTRUSION: {alerts}" if alerts else ""))

        annotated = draw_frame(frame, zones, tracks, records, alert_frames, fps)
        if single_image:
            last_annotated = annotated
        elif args.frames:
            cv2.imwrite(str(out_dir / f"{Path(name).stem}.jpg"), annotated)
        else:
            writer.write(annotated)

    if single_image:
        cv2.imwrite(args.output, last_annotated)
    elif not args.frames:
        writer.release()

    intruder_ids = [tid for tid, rec in records.items() if rec.ever_intrusion(alert_frames)]
    print(f"[result] {len(records)} distinct person(s) tracked across {len(frames)} frame(s), "
          f"{len(intruder_ids)} flagged INTRUSION: {sorted(intruder_ids)}")
    for tid in sorted(intruder_ids):
        print(f"  ID {tid}: {records[tid].dwell_seconds(fps):.1f}s in {records[tid].last_zone_name}")
    print(f"[done  ] wrote {args.output}")
