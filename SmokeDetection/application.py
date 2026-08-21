"""Multi-frame sanity test for smoke (and fire) detection.

Model search: Hugging Face has several community smoke detectors.
kittendev/YOLOv8m-smoke-detection (single class "smoke", claimed
mAP@0.5=0.995) was tried first since it's a dedicated smoke-only model
with real community adoption (196 downloads, 19 likes). Rejected on
inspection: its own thumbnail.jpg (the model card's own demo grid) shows
the same handful of smoke shapes digitally composited onto random stock
photos (beaches, bathrooms, living rooms) -- i.e. synthetic overlay data,
not photographed smoke. This showed up directly in testing: on three real
photos with obvious, unmistakable smoke (an industrial chimney, a dense
smokestack plume, a forest fire) it scored 0.03-0.19 confidence -- below
any usable threshold, meaning it doesn't generalize past its own
synthetic-composite training distribution.

Switched to rabahdev/fire-smoke-yolov8n (YOLOv8n fine-tuned on D-Fire, a
professional 14k-train-image real-photo fire/smoke dataset, 2 classes:
smoke, fire; realistic reported mAP50=0.754, not suspiciously perfect).
Re-ran the same three real photos: chimney smoke 0.455 (smoke), dense
smokestack plume 0.859 (smoke), forest fire 0.826 (fire) -- properly
differentiated, confident, correct-class results. Cross-checked against
three genuinely smoke-free photos (an unrelated factory floor, a
concrete barrier, a construction site): 0.000-0.006 confidence -- clean
negatives, not a detector that fires on everything. Ships only best.pt on
HF, so exported once with `YOLO(weights).export(format="onnx", opset=12,
simplify=True)` -> model/fire_smoke_yolov8n.onnx.

Although the task is "Smoke Detection", this model's fire class is kept
and surfaced too (not discarded) -- it's the same detector, the same
verified reliability, and a fire detection is at least as safety-critical
as a smoke one for anyone deploying this. The banner/alert logic treats
both classes identically.

Detector I/O: standard Ultralytics YOLOv8n export, nc=2. Input
images[1,3,640,640] RGB letterboxed, /255. Output [1,6,8400] = 4 box
(cx,cy,w,h, pixel-space, already decoded) + 2 class scores (already
sigmoid'd, same convention verified in every other YOLO detector in this
project). Needs transpose + confidence filter + NMS.

Smoke/fire has no stable "identity" the way a person does (a plume grows,
splits, diffuses), but the same lightweight SORT-style tracker (IOU +
Hungarian, from ContainerTracking/CustomerDwellTime/TheftDetection/
RestrictedAreaMonitoring) still works well enough frame-to-frame to give
each detected region a persistent-ish ID and require --alert-frames
consecutive detections before raising ALERT -- guards against a single
flickery false-positive frame triggering an alarm. A single image is also
accepted as a lighter sanity check (see --input docs).
"""

import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from scipy.optimize import linear_sum_assignment

CLASS_NAMES = ["smoke", "fire"]
DEFAULT_FPS_FALLBACK = 25.0

# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIRS = ("model", "models", "weights", "onnx", "checkpoints", ".")
# rabahdev/fire-smoke-yolov8n ships only best.pt on HF (no .onnx), so
# auto-fetch can't resolve it directly -- the exported model/
# fire_smoke_yolov8n.onnx is committed locally instead (see model/download.sh).
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
class SmokeFireDetector:
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
        boxes_xywh, scores = preds[:, :4], preds[:, 4:4 + len(CLASS_NAMES)]
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
# (same approach as ContainerTracking/CustomerDwellTime/TheftDetection/
# RestrictedAreaMonitoring).
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


class Track:
    def __init__(self, track_id, detection):
        self.id = track_id
        self.box = detection["box"]
        self.class_name = detection["class"]
        self.score = detection["score"]
        self.misses = 0
        self.hits = 1

    def update(self, detection):
        self.box = detection["box"]
        self.class_name = detection["class"]
        self.score = detection["score"]
        self.hits += 1
        self.misses = 0


class RegionTracker:
    def __init__(self, iou_threshold=0.2, max_age=10):
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
class AlertRecord:
    """Per-track-ID bookkeeping: consecutive-frame streak (for alert
    debounce) plus cumulative seconds detected (how long this smoke/fire
    has been visible). Kept independent of the tracker's own track list so
    stats survive after a track is pruned."""

    def __init__(self):
        self.streak = 0
        self.frames_seen = 0
        self.class_name = None

    def update(self, class_name):
        self.frames_seen += 1
        self.streak += 1
        self.class_name = class_name

    def miss(self):
        self.streak = 0

    def is_alert(self, alert_frames):
        return self.streak >= alert_frames

    def duration_seconds(self, fps):
        return self.frames_seen / fps


CLASS_COLOR = {"smoke": (180, 180, 180), "fire": (0, 100, 255)}


def draw_frame(image, tracks, records, alert_frames, fps):
    out = image.copy()
    h, w = out.shape[:2]
    scale = w / 1400
    box_thick = max(2, int(3 * scale))
    label_scale = max(0.8, 1.1 * scale)
    label_thick = max(1, int(2 * scale))

    active = [t for t in tracks if t.misses == 0]
    alert_count = sum(1 for t in active if records[t.id].is_alert(alert_frames))

    # Reserve the banner's vertical strip BEFORE placing any per-detection
    # label -- a detection box near the top of the frame (e.g. a smoke
    # plume that fills most of the image) would otherwise get its own
    # label drawn right where the banner goes, and the two collide/overlap
    # since the banner isn't guaranteed wider than the label.
    banner = f"SMOKE/FIRE ALERTS: {alert_count}"
    banner_scale = max(0.8, 1.1 * scale)
    banner_thick = max(2, int(3 * scale))
    (btw, bth), bbaseline = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, banner_scale, banner_thick)
    bpad = int(10 * scale) + 6
    banner_bottom = bth + bbaseline + bpad * 2

    # Boxes whose top edge falls inside the reserved banner strip all need
    # their label pushed down -- stack them below each other (in track-ID
    # order) instead of all landing on the same forced height, which would
    # just move the collision from "label vs banner" to "label vs label"
    # (happens often here since smoke plumes/flame regions commonly span
    # most of the frame, unlike a person's much smaller box).
    stack_y = banner_bottom
    for trk in sorted(active, key=lambda t: t.id):
        rec = records[trk.id]
        alert = rec.is_alert(alert_frames)
        color = (0, 0, 255) if alert else CLASS_COLOR.get(trk.class_name, (0, 255, 255))

        x1, y1, x2, y2 = map(int, trk.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thick)
        label = f"{trk.class_name.upper()} {trk.score:.2f}" + (f"  ALERT {rec.duration_seconds(fps):.1f}s" if alert else "")
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
    """Returns (iterable of (name, frame), fps, frame_count).

    Video frames are yielded lazily rather than accumulated in a list: a
    decoded 1080p frame is ~6 MB, so buffering a whole clip costs ~6 MB *
    fps * seconds (a 10-min 30 fps clip is >100 GB) and dies with
    MemoryError long before the detector ever runs. Only one frame is held
    at a time now. Image/directory inputs stay eager -- they're small and
    the count has to be known up front.

    frame_count is always the exact number of frames the caller will get,
    for every input kind -- it's printed before decoding starts and decides
    whether the source is empty, so an approximation won't do.
    """
    p = Path(source)
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(source)
        return [(p.stem, img)], None, 1

    if p.is_dir():
        paths = sorted(p.glob("*.jpg")) + sorted(p.glob("*.png"))
        return [(str(fp), cv2.imread(str(fp))) for fp in paths], None, len(paths)

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise FileNotFoundError(f"Not an image, directory, or readable video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or None

    # CAP_PROP_FRAME_COUNT is container metadata, not a decode result, and it
    # regularly disagrees with the frames that actually come out: measured
    # here, .mkv/.webm/.ts/.avi all over-report (60/59 claimed vs 50 decoded)
    # and a raw .h264 stream reports -192153584101141. Printing that in the
    # [input ] line would contradict the [result] line of the same run, so
    # count for real with a grab-only pass first. grab() skips the decode-to-
    # array step, so it costs ~1% of the detector's own runtime (0.24s for 300
    # 720p frames) and, unlike the list this replaced, holds no frames at all.
    count = 0
    while cap.grab():
        count += 1
    cap.release()

    def stream():
        cap2 = cv2.VideoCapture(str(p))
        idx = 0
        try:
            while True:
                ok, frame = cap2.read()
                if not ok:
                    break
                yield (f"frame_{idx:04d}", frame)
                idx += 1
        finally:
            cap2.release()

    return stream(), fps, count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smoke/fire detection sanity test (image, video, or frame directory in).")
    parser.add_argument("-i", "--input", required=True,
                        help="Path to a single image, a video file, OR a directory of ordered "
                             "frame images.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path: an image for single-image input, a video for video/"
                             "frame-dir input (or a directory if --frames is set). "
                             "Default: output.jpg / smoke_tracked.mp4")
    parser.add_argument("-m", "--model", default=None,
                        help="Path to .onnx, a folder containing one, or a HF ref. "
                             "Omit to auto-discover in: " + ", ".join(DEFAULT_MODEL_DIRS))
    parser.add_argument("-c", "--conf-threshold", type=float, default=0.35)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--alert-frames", type=int, default=3,
                         help="Consecutive frames smoke/fire must be detected in the same tracked "
                              "region before an ALERT is raised (default: 3)")
    parser.add_argument("--iou-threshold", type=float, default=0.2,
                         help="Minimum IOU for the tracker to match a detection to an existing "
                              "track (lower than the person-tracking use cases -- a smoke plume's "
                              "box shifts/grows more between frames than a walking person's does)")
    parser.add_argument("--max-age", type=int, default=10,
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
        args.output = "output.jpg" if single_image else "smoke_tracked.mp4"
    alert_frames = 1 if single_image else args.alert_frames

    detector = SmokeFireDetector(resolve_model_path(args.model))
    tracker = RegionTracker(iou_threshold=args.iou_threshold, max_age=args.max_age)

    frames, detected_fps, frame_count = load_frames(args.input)
    if frame_count == 0:
        raise FileNotFoundError(f"No frames found at {args.input}")
    fps = args.fps or detected_fps or DEFAULT_FPS_FALLBACK
    print(f"[input ] {frame_count} frame(s) from {args.input} @ {fps:.1f} fps")
    if single_image:
        print("[note  ] single-image input -- showing the raw per-frame detection, no "
              "multi-frame alert debounce (run on a video for that)")

    writer = None  # opened lazily on the first frame -- with streaming input the
                   # frame size isn't known until one has actually been decoded
    if single_image:
        pass  # single annotated frame is written directly after the loop, no writer needed
    elif args.frames:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

    records = {}
    last_annotated = None
    n_frames = 0
    for frame_idx, (name, frame) in enumerate(frames):
        n_frames += 1
        detections = detector.detect(frame, confidence_threshold=args.conf_threshold,
                                      nms_threshold=args.nms_threshold)
        tracks = tracker.update(detections)

        active = [t for t in tracks if t.misses == 0]
        active_ids = set()
        for trk in active:
            if trk.id not in records:
                records[trk.id] = AlertRecord()
            records[trk.id].update(trk.class_name)
            active_ids.add(trk.id)
        for tid, rec in records.items():
            if tid not in active_ids:
                rec.miss()

        alerts = [t.id for t in active if records[t.id].is_alert(alert_frames)]
        print(f"  {name}: {len(active)} detection(s) -> "
              f"{[f'{t.class_name}(ID{t.id},{t.score:.2f})' for t in active]}"
              + (f"  ALERT: {alerts}" if alerts else ""))

        annotated = draw_frame(frame, tracks, records, alert_frames, fps)
        if single_image:
            last_annotated = annotated
        elif args.frames:
            cv2.imwrite(str(out_dir / f"{Path(name).stem}.jpg"), annotated)
        else:
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                         min(fps, 30), (w, h))
            writer.write(annotated)

    if n_frames == 0:  # source emptied out between the count and the read
        raise FileNotFoundError(f"No frames found at {args.input}")

    if single_image:
        cv2.imwrite(args.output, last_annotated)
    elif not args.frames:
        writer.release()

    alerted_ids = [tid for tid, rec in records.items() if rec.is_alert(alert_frames)]
    print(f"[result] {len(records)} distinct region(s) tracked across {n_frames} frame(s), "
          f"{len(alerted_ids)} alerted: {sorted(alerted_ids)}")
    for tid in sorted(alerted_ids):
        print(f"  ID {tid}: {records[tid].class_name} for {records[tid].duration_seconds(fps):.1f}s")
    print(f"[done  ] wrote {args.output}")
