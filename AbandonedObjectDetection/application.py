"""Multi-frame sanity test for abandoned/unattended object detection
(Surveillance > Security).

Searched Hugging Face/GitHub for a dedicated "abandoned object"/
"unattended luggage" model -- every real system found (e.g. the
YOLO11-based airport-luggage projects on Hackster.io, and the academic
SAO-YOLO / dual-background papers) uses the same architecture rather than
a single trained "abandoned" class: a generic object detector + a
multi-object tracker + an owner-association rule on top, exactly the
zone/geometry-rule pattern already used in this project for
QueueLengthMonitoring/CustomerDwellTime/RestrictedAreaMonitoring/
LoiteringDetection. So this reuses the already-verified Ultralytics
YOLO11s (COCO-pretrained) detector from those same use cases -- COCO-80
already includes "backpack"/"handbag"/"suitcase" alongside "person", no
custom training needed -- and the SORT-style IOU+Hungarian tracker from
ContainerTracking/CustomerDwellTime/RestrictedAreaMonitoring/
LoiteringDetection, this time tracking the LUGGAGE, not people.

What's specific to this use case: an object is "abandoned" only when BOTH
of these hold at once, continuously:
  - STATIONARY: its tracked position barely moved since the previous
    frame (displacement < 20% of its own box diagonal -- scaled to the
    object's own size rather than a fixed pixel count, since the same
    displacement means something different for a bag filling half the
    frame vs. a distant one). A brand-new track has no prior position, so
    it counts as stationary from frame 1 -- if it's also unattended, the
    clock starts immediately, which matches how a real system would
    treat "first ever saw this bag, and no one's next to it."
  - UNATTENDED: no currently-detected person's box is within a
    size-relative distance (3x the object's own box diagonal) of it, box
    edge to box edge (0 if they already overlap). Deliberately measured
    box-to-box rather than to a person's foot-point or box center: a
    backpack worn on someone's back overlaps their box directly but sits
    at shoulder height, far from their foot-point (and often far from
    their box center too, in a full-body shot) -- an earlier version of
    this script used foot-point distance and consequently mislabeled a
    worn backpack as unattended, caught by testing samples/sample_attended
    .jpg before shipping. Box-to-box is also a deliberately simple
    stand-in for "owner association" -- real systems in the literature
    link a specific person to a specific bag over time before it's
    dropped; this just checks "is anyone plausibly close enough to be its
    owner right now", which is enough for a sanity check but will
    false-positive if a stranger happens to walk near someone else's
    stationary bag.
Both conditions must hold in the SAME frame, continuously, for
--abandoned-seconds before ABANDONED fires (default 15.0s) -- moving OR
being approached by a person resets the clock, same continuous-session
philosophy as LoiteringDetection (as opposed to a cumulative, gap-
tolerant timer).

Verified on two real photos: samples/sample_unattended.jpg (a suitcase +
backpack alone outdoors, no person anywhere in frame) correctly shows
every detected bag as unattended; samples/sample_attended.jpg (a hiker
photographed from behind, backpack worn directly on his back) correctly
shows the backpack as attended, since the nearest person distance is ~0.
A single image is one frame -- it can't accumulate real seconds, so
--abandoned-seconds is forced to 0 for single-image input, meaning the
demo really only exercises the unattended/attended geometry, not true
motion-stationarity (see --input docs); run on a video for a real
time-threshold reading.

Detector I/O: identical to the sibling use cases (input images[1,3,640,640]
RGB letterboxed, /255; output [1,84,8400] raw 4-box+80-class COCO, needs
transpose + confidence filter + NMS).
"""

import argparse
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
LUGGAGE_CLASSES = {"backpack", "handbag", "suitcase"}
DEFAULT_FPS_FALLBACK = 25.0

# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIRS = ("model", "models", "weights", "onnx", "checkpoints", ".")
DEFAULT_HF_REPO = None  # Ultralytics/YOLO11 ships only .pt; exported model/yolo11s.onnx is committed locally


def resolve_model_path(explicit=None, search_dirs=DEFAULT_MODEL_DIRS):
    if explicit:
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
        raise FileNotFoundError("No .onnx found in: " + ", ".join(str(d) for d in search_dirs))
    if len(candidates) > 1:
        listing = "\n  ".join(f"{c}  ({c.stat().st_size / 1e6:.1f} MB)" for c in candidates)
        raise RuntimeError(f"Found {len(candidates)} .onnx files — pick one with -m:\n  {listing}")
    print(f"[model] auto-resolved: {candidates[0]}")
    return str(candidates[0])


# ---------------------------------------------------------------------------
class ObjectDetector:
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
# (same approach as ContainerTracking/CustomerDwellTime/RestrictedAreaMonitoring/
# LoiteringDetection). Tracks luggage only -- people are used per-frame, untracked.
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


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def box_distance(box_a, box_b):
    """0 if the boxes overlap or touch, else the gap between their nearest
    edges. Used instead of center- or foot-point-to-center distance: a worn
    backpack sits at shoulder height, nowhere near the wearer's foot-point
    or box center in a full-body shot, but its box still overlaps theirs."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return float(np.hypot(dx, dy))


def diagonal(box):
    x1, y1, x2, y2 = box
    return float(np.hypot(x2 - x1, y2 - y1))


class Track:
    def __init__(self, track_id, detection):
        self.id = track_id
        self.box = detection["box"]
        self.class_name = detection["class"]
        self.score = detection["score"]
        self.misses = 0
        self.hits = 1
        self.centers = [center(self.box)]

    def update(self, detection):
        self.box = detection["box"]
        self.class_name = detection["class"]
        self.score = detection["score"]
        self.hits += 1
        self.misses = 0
        self.centers.append(center(self.box))
        self.centers = self.centers[-30:]

    def displacement(self):
        if len(self.centers) < 2:
            return 0.0
        (x1, y1), (x2, y2) = self.centers[-2], self.centers[-1]
        return float(np.hypot(x2 - x1, y2 - y1))


class LuggageTracker:
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
class AbandonRecord:
    """Per-track-ID bookkeeping: consecutive-frame streak where the object
    was BOTH stationary and unattended in the same frame (resets to 0 the
    moment either condition breaks). Kept independent of the tracker's own
    track list so stats survive after a track is pruned."""

    def __init__(self):
        self.streak = 0
        self.stationary = False
        self.unattended = False
        self.nearest_person_dist = None

    def update(self, stationary, unattended, nearest_person_dist):
        self.stationary = stationary
        self.unattended = unattended
        self.nearest_person_dist = nearest_person_dist
        if stationary and unattended:
            self.streak += 1
        else:
            self.streak = 0

    def session_seconds(self, fps):
        return self.streak / fps

    def is_abandoned(self, abandoned_seconds, fps):
        # streak > 0 guards the single-image case, where --abandoned-seconds
        # is forced to 0.0 -- without it, `0.0 >= 0.0` would be true even
        # for a streak of 0 (i.e. a correctly-attended object).
        return self.streak > 0 and self.session_seconds(fps) >= abandoned_seconds


# ---------------------------------------------------------------------------
def draw_frame(image, tracks, records, abandoned_seconds, fps):
    out = image.copy()
    h, w = out.shape[:2]
    scale = w / 1400
    box_thick = max(2, int(3 * scale))
    label_scale = max(0.8, 1.1 * scale)
    label_thick = max(1, int(2 * scale))

    active = [t for t in tracks if t.misses == 0]
    alert_count = sum(1 for t in active if records[t.id].is_abandoned(abandoned_seconds, fps))

    banner = f"TRACKED OBJECTS: {len(active)}   ABANDONED: {alert_count}"
    banner_scale = max(0.8, 1.1 * scale)
    banner_thick = max(2, int(3 * scale))
    (btw, bth), bbaseline = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, banner_scale, banner_thick)
    bpad = int(10 * scale) + 6
    banner_bottom = bth + bbaseline + bpad * 2

    stack_y = banner_bottom
    for trk in sorted(active, key=lambda t: t.box[1]):
        rec = records[trk.id]
        alert = rec.is_abandoned(abandoned_seconds, fps)
        color = (0, 0, 255) if alert else (0, 140, 255) if rec.unattended else (0, 200, 0)

        x1, y1, x2, y2 = map(int, trk.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thick)
        status = f"ABANDONED! {rec.session_seconds(fps):.1f}s" if alert else \
            ("UNATTENDED" if rec.unattended else "ATTENDED")
        label = f"{trk.class_name} ID{trk.id}  {status}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thick)
        needed = th + baseline + 8
        if y1 < stack_y + needed:
            label_y = stack_y + needed
            stack_y = label_y + int(4 * scale)
        else:
            label_y = y1
        cv2.rectangle(out, (x1, label_y - th - baseline - 8), (x1 + tw + 12, label_y), color, -1)
        cv2.putText(out, label, (x1 + 6, label_y - baseline - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, label_scale, (0, 0, 0), label_thick, cv2.LINE_AA)

    cv2.rectangle(out, (0, 0), (min(w, btw + bpad * 2), banner_bottom), (0, 0, 0), -1)
    color = (0, 0, 255) if alert_count else (0, 255, 0)
    cv2.putText(out, banner, (bpad, bth + bpad), cv2.FONT_HERSHEY_SIMPLEX, banner_scale, color, banner_thick, cv2.LINE_AA)
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
        description="Abandoned/unattended object detection sanity test (image, video, or frame directory in).")
    parser.add_argument("-i", "--input", required=True,
                        help="Path to a single image, a video file, OR a directory of ordered "
                             "frame images.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path: an image for single-image input, a video for video/"
                             "frame-dir input (or a directory if --frames is set). "
                             "Default: output.jpg / abandoned_tracked.mp4")
    parser.add_argument("-m", "--model", default=None,
                        help="Path to .onnx or a folder containing one. "
                             "Omit to auto-discover in: " + ", ".join(DEFAULT_MODEL_DIRS))
    parser.add_argument("-c", "--conf-threshold", type=float, default=0.35)
    parser.add_argument("--abandoned-seconds", type=float, default=15.0,
                         help="Consecutive seconds an object must be simultaneously stationary "
                              "and unattended before an ABANDONED alert is raised (default: 15.0)")
    parser.add_argument("--movement-frac", type=float, default=0.2,
                         help="Frame-to-frame center displacement, as a fraction of the object's "
                              "own box diagonal, below which it's considered stationary this frame")
    parser.add_argument("--attend-radius-frac", type=float, default=3.0,
                         help="A person whose box is within this many box-diagonals of the "
                              "object (edge to edge, 0 if overlapping) counts as attending it")
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
        args.output = "output.jpg" if single_image else "abandoned_tracked.mp4"
    # A single frame is a tiny fraction of a second and can never cross a
    # real multi-second threshold -- force it to 0 so the ABANDONED! styling
    # is still demonstrated (same rationale as LoiteringDetection).
    abandoned_seconds = 0.0 if single_image else args.abandoned_seconds

    detector = ObjectDetector(resolve_model_path(args.model))
    tracker = LuggageTracker(iou_threshold=args.iou_threshold, max_age=args.max_age)

    frames, detected_fps = load_frames(args.input)
    if not frames:
        raise FileNotFoundError(f"No frames found at {args.input}")
    fps = args.fps or detected_fps or DEFAULT_FPS_FALLBACK
    print(f"[input ] {len(frames)} frame(s) from {args.input} @ {fps:.1f} fps")
    if single_image:
        print("[note  ] single-image input -- --abandoned-seconds forced to 0, so only the "
              "attended/unattended geometry is exercised (run on a video for real motion + "
              "time-threshold behavior)")

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
        luggage_dets = [d for d in detections if d["class"] in LUGGAGE_CLASSES]
        person_dets = [d for d in detections if d["class"] == PERSON_CLASS]
        tracks = tracker.update(luggage_dets)

        active = [t for t in tracks if t.misses == 0]
        for trk in active:
            if trk.id not in records:
                records[trk.id] = AbandonRecord()
            diag = diagonal(trk.box)
            stationary = trk.displacement() < args.movement_frac * diag
            if person_dets:
                nearest = min(box_distance(trk.box, p["box"]) for p in person_dets)
            else:
                nearest = float("inf")
            unattended = nearest > args.attend_radius_frac * diag
            records[trk.id].update(stationary, unattended, nearest)

        alerts = [t.id for t in active if records[t.id].is_abandoned(abandoned_seconds, fps)]
        print(f"  {name}: {len(active)} object(s) -> "
              f"{[f'{t.class_name}(ID{t.id})' for t in active]}"
              + (f"  ABANDONED: {alerts}" if alerts else ""))

        annotated = draw_frame(frame, tracks, records, abandoned_seconds, fps)
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

    abandoned_ids = [tid for tid, rec in records.items() if rec.is_abandoned(abandoned_seconds, fps)]
    print(f"[result] {len(records)} distinct object(s) tracked across {len(frames)} frame(s), "
          f"{len(abandoned_ids)} flagged ABANDONED: {sorted(abandoned_ids)}")
    for tid in sorted(abandoned_ids):
        print(f"  ID {tid}: {records[tid].session_seconds(fps):.1f}s stationary+unattended")
    print(f"[done  ] wrote {args.output}")
