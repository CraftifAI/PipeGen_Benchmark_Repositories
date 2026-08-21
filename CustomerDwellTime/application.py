"""Multi-frame sanity test for customer dwell time monitoring.

Like tracking (see ContainerTracking) and queue length (see
QueueLengthMonitoring), dwell time cannot be shown on a single image at
all -- it's specifically "how long has THIS SAME PERSON stayed inside a
zone", which needs a persistent ID across frames plus a clock. Takes a
video (or a directory of ordered frame images) instead of one image.

No dedicated "dwell time" model exists: dwell time is person detection +
tracking + zone-membership + elapsed-time bookkeeping, not a trained
visual class (same situation as QueueLengthMonitoring/DockUtilization/
ShelfOccupancy). Reuses the already-verified Ultralytics YOLO11s
(COCO-pretrained) person detector from VehicleDetection/DockUtilization/
QueueLengthMonitoring/CustomerHeatmaps, a minimal from-scratch SORT-style
tracker (IOU + Hungarian matching, same as ContainerTracking -- multi-
object tracking is a classical algorithm, not something to search for on
Hugging Face), and a configurable zone polygon (same --zone-file
convention as QueueLengthMonitoring).

Dwell time definition used here: for each tracked person, the cumulative
number of seconds their foot-point (bottom-center of their box) has been
inside the zone across the whole time they're tracked, converted from
frame counts using the source video's fps (or --fps if the container
reports none/an unreliable value). This is the standard actionable
retail-analytics metric (feeds directly into "customers lingering near
this display" alerts) -- it does not attempt cross-session
re-identification, so if a track is lost (e.g. long occlusion past
--max-age) and the same person reappears, they start a new ID and a new
dwell clock, same limitation ContainerTracking documents for its tracker.

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
# Ultralytics/YOLO11 ships only .pt on HF, so auto-fetch can't resolve it
# directly -- export locally with:
#   python -c "from ultralytics import YOLO; YOLO('yolo11s.pt').export(format='onnx', opset=12, simplify=True)"
# and place the result at model/yolo11s.onnx (see model/download.sh).
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
# Minimal SORT-style tracker: IOU + Hungarian matching, no appearance model
# (same approach as ContainerTracking).
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


class Track:
    def __init__(self, track_id, detection):
        self.id = track_id
        self.box = detection["box"]
        self.misses = 0
        self.hits = 1
        self.trail = [self._center()]

    def _center(self):
        x1, y1, x2, y2 = self.box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def update(self, detection):
        self.box = detection["box"]
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
def default_zone(img_shape):
    """Whole frame -- used when no --zone-file is given. A real deployment
    would draw a zone matching the actual area of interest (a display, an
    aisle, a checkout counter) once for a fixed camera."""
    h, w = img_shape[:2]
    return [[0, 0], [w, 0], [w, h], [0, h]]


def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.float32), point, False) >= 0


class DwellRecord:
    """Cumulative zone-dwell bookkeeping for one track ID, kept independent
    of the tracker's own track list so stats survive after a track is
    pruned (e.g. the person walked off-frame)."""

    def __init__(self, first_frame):
        self.first_frame = first_frame
        self.last_frame = first_frame
        self.frames_in_zone = 0
        self.in_zone = False
        self.entry_frame = None  # frame the CURRENT continuous zone stay started

    def update(self, frame_idx, in_zone_now):
        self.last_frame = frame_idx
        if in_zone_now:
            self.frames_in_zone += 1
            if not self.in_zone:
                self.entry_frame = frame_idx
        self.in_zone = in_zone_now

    def dwell_seconds(self, fps):
        return self.frames_in_zone / fps

    def current_session_seconds(self, frame_idx, fps):
        if not self.in_zone or self.entry_frame is None:
            return 0.0
        return (frame_idx - self.entry_frame + 1) / fps


# ---------------------------------------------------------------------------
def color_for_id(track_id):
    rng = np.random.default_rng(track_id * 7919)
    return tuple(int(c) for c in rng.integers(80, 255, size=3))


def draw_frame(image, tracks, dwell_records, zone, frame_idx, fps, dwell_threshold):
    out = image.copy()
    h, w = out.shape[:2]
    scale = max(w, h) / 1400
    box_thick = max(2, int(3 * scale))
    label_scale = max(0.8, 1.1 * scale)
    label_thick = max(1, int(2 * scale))

    overlay = out.copy()
    cv2.fillPoly(overlay, [np.array(zone, dtype=np.int32)], (255, 0, 255))
    out = cv2.addWeighted(overlay, 0.12, out, 0.88, 0)
    cv2.polylines(out, [np.array(zone, dtype=np.int32)], True, (255, 0, 255), max(2, int(3 * scale)))

    in_zone_count, lingering_count, longest = 0, 0, (None, 0.0)
    for trk in tracks:
        if trk.misses > 0:
            continue
        rec = dwell_records[trk.id]
        dwell_s = rec.dwell_seconds(fps)
        if rec.in_zone:
            in_zone_count += 1
            if dwell_s > longest[1]:
                longest = (trk.id, dwell_s)
        lingering = rec.in_zone and dwell_s >= dwell_threshold
        lingering_count += int(lingering)

        color = (0, 0, 255) if lingering else (0, 200, 255) if rec.in_zone else (150, 150, 150)
        x1, y1, x2, y2 = map(int, trk.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thick)
        fx, fy = map(int, foot_point(trk.box))
        cv2.circle(out, (fx, fy), max(3, int(5 * scale)), color, -1)
        for i in range(1, len(trk.trail)):
            cv2.line(out, trk.trail[i - 1], trk.trail[i], color, max(1, box_thick // 2))

        label = f"ID {trk.id}  {dwell_s:.1f}s" + ("  LINGER" if lingering else "")
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thick)
        label_y = max(th + baseline + 8, y1)
        cv2.rectangle(out, (x1, label_y - th - baseline - 8), (x1 + tw + 12, label_y), color, -1)
        cv2.putText(out, label, (x1 + 6, label_y - baseline - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, label_scale, (0, 0, 0), label_thick, cv2.LINE_AA)

    banner = f"IN ZONE: {in_zone_count}   LINGERING (>={dwell_threshold:.0f}s): {lingering_count}"
    if longest[0] is not None:
        banner += f"   LONGEST: ID {longest[0]} {longest[1]:.1f}s"
    banner_scale = max(1.0, 1.3 * scale)
    banner_thick = max(2, int(3 * scale))
    (tw, th), baseline = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, banner_scale, banner_thick)
    pad = int(12 * scale) + 8
    cv2.rectangle(out, (0, 0), (min(w, tw + pad * 2), th + baseline + pad * 2), (0, 0, 0), -1)
    color = (0, 0, 255) if lingering_count > 0 else (0, 255, 0)
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
        # every extension IMAGE_EXTS advertises, in ONE name order -- globbing
        # per-extension would replay all .jpg frames before all .png frames and
        # shatter the tracker (and silently miss .jpeg/.bmp/.webp frame dirs).
        paths = sorted(fp for fp in p.iterdir()
                       if fp.is_file() and fp.suffix.lower() in IMAGE_EXTS)
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
        description="Customer dwell time monitoring sanity test (image, video, or frame directory in).")
    parser.add_argument("-i", "--input", required=True,
                        help="Path to a single image, a video file, OR a directory of ordered "
                             "frame images. A single image only shows the FIRST instant a person "
                             "is seen (dwell time is inherently a multi-frame measurement) -- pass "
                             "a video for a real dwell-time reading.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path: an image for single-image input, a video for video/"
                             "frame-dir input (or a directory if --frames is set). "
                             "Default: output.jpg / dwell_tracked.mp4")
    parser.add_argument("-m", "--model", default=None,
                        help="Path to .onnx, a folder containing one, or a HF ref. "
                             "Omit to auto-discover in: " + ", ".join(DEFAULT_MODEL_DIRS))
    parser.add_argument("-c", "--conf-threshold", type=float, default=0.35)
    parser.add_argument("--zone-file", default=None,
                         help="JSON file: list of [x,y] polygon points marking the area of "
                              "interest (e.g. a display or aisle). Omit to use the whole frame.")
    parser.add_argument("--dwell-threshold", type=float, default=10.0,
                         help="Seconds a person must accumulate inside the zone before being "
                              "flagged as 'lingering' (default: 10.0)")
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
        args.output = "output.jpg" if single_image else "dwell_tracked.mp4"

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
        print("[note  ] single-image input -- dwell times below are just the first instant each "
              "person is seen; run on a video for real dwell-time tracking")

    if args.zone_file:
        zone = json.load(open(args.zone_file))
        print(f"[zone  ] loaded from {args.zone_file}")
    else:
        zone = default_zone(frames[0][1].shape)
        print("[zone  ] no --zone-file given, using whole frame")

    if single_image:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    elif args.frames:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        h, w = frames[0][1].shape[:2]
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), min(fps, 30), (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter could not open {args.output} (mp4v, {w}x{h})")

    dwell_records = {}
    last_annotated = None
    for frame_idx, (name, frame) in enumerate(frames):
        detections = detector.detect(frame, confidence_threshold=args.conf_threshold)
        people = [d for d in detections if d["class"] == PERSON_CLASS]
        tracks = tracker.update(people)

        active = [t for t in tracks if t.misses == 0]
        for trk in active:
            if trk.id not in dwell_records:
                dwell_records[trk.id] = DwellRecord(frame_idx)
            in_zone_now = point_in_polygon(foot_point(trk.box), zone)
            dwell_records[trk.id].update(frame_idx, in_zone_now)

        print(f"  {name}: {len(people)} person(s) -> "
              f"{[f'ID{t.id}({dwell_records[t.id].dwell_seconds(fps):.1f}s)' for t in active]}")

        annotated = draw_frame(frame, tracks, dwell_records, zone, frame_idx, fps, args.dwell_threshold)
        if single_image:
            last_annotated = annotated
        elif args.frames:
            cv2.imwrite(str(out_dir / f"{Path(name).stem}.jpg"), annotated)
        else:
            writer.write(annotated)

    if single_image:
        if not cv2.imwrite(args.output, last_annotated):
            raise RuntimeError(f"cv2.imwrite failed for {args.output}")
    elif not args.frames:
        writer.release()

    print(f"[result] {len(dwell_records)} distinct person(s) tracked across {len(frames)} frame(s)")
    for track_id, rec in sorted(dwell_records.items(), key=lambda kv: kv[1].dwell_seconds(fps), reverse=True):
        flag = "  LINGERED" if rec.dwell_seconds(fps) >= args.dwell_threshold else ""
        print(f"  ID {track_id}: {rec.dwell_seconds(fps):.1f}s in zone{flag}")
    print(f"[done  ] wrote {args.output}")
