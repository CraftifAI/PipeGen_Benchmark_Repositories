"""Multi-frame sanity test for retail theft (shoplifting) detection.

Model search: no GitHub repo ships downloadable weights for this (mostly
research papers, or CCTV-fight/violence detectors that don't cover
shoplifting specifically). On Hugging Face, searched "shoplifting",
"theft detection", "shop lifting", "suspicious activity detection":
  - Several VideoMAE/ViViT video-classification fine-tunes exist (e.g.
    Abdullah1/videomae-..., yehiawp4/ViViT-b-16x2-ShopLifting-Dataset).
    Rejected: these need a fixed-length clip (8-32 frames) through a
    transformer backbone -- a fundamentally heavier pipeline than every
    other use case in this project, and none published an ONNX export.
  - Vaibhavsh0120/ATM-Theft-Detection (YOLOv8, "security"/"atm-monitoring"
    tags): rejected on inspection -- despite the repo name, its only two
    classes are Face_Covered/Face_Uncovered, i.e. a face-covering
    detector, not theft.
  - Accurateinfosolution/Suspicious_activity_detection_Yolov11_Custom: a
    10-class YOLO (Assault, Fighting, Gun, Kidnapping, Knife, People,
    Police, Prisoner, Theft/Robbery, Time Bomb) with a Theft/Robbery
    class, but it's a broad crime-scene detector (weapons, uniforms) with
    no published validation -- a worse match than a model trained
    specifically on shoplifting.
  - Nour190/shoplifting-yolo: a YOLOv8s-Pose model (ships best.pt only)
    trained on a Roboflow "Shoplifting-Detection-1" dataset with exactly
    2 classes, {0: normal, 1: shoplifting} -- the most direct match found.
    Exported once with `YOLO(weights).export(format="onnx", opset=12,
    simplify=True)` (mechanical, not training) -> model/
    shoplifting_yolov8s_pose.onnx.

Verification done: run against a real, in-domain photo (a person walking
an eye-level grocery-store aisle, full body visible -- samples/sample.jpg)
-- correctly scored normal=0.95, shoplifting=0.0001. Sanity-checked the
classifier isn't degenerate (always-0/always-same) by running it across
six unrelated photos and observing the shoplifting score vary from 0.00 to
0.62 depending on image content.

Positive-class check: free stock photo sites have essentially no genuine
staged-theft photos, so a real 44-frame CCTV clip was sourced instead from
1amitos1/Shoplifting-Detection's own DB_Sample/input/ (raw, unannotated
frames that project collected from real supermarket security cameras for
its own shoplifting-detection research -- samples/theft_clip.mp4). Run
end-to-end through this exact script: the SAME tracked person scored
normal (0.85-0.95) for the first 35 frames of ordinary browsing, then the
model switched to shoplifting (0.66-0.91) for 8 straight frames right as
the person crouches down at a low shelf (samples/sample_theft.jpg /
samples/output_theft.jpg is frame 39 of that transition), before
reverting to normal in the last frame -- exactly the sustained,
context-plausible transition a real detector should show, not single-
frame noise. This is the strongest evidence available that the
shoplifting class is a real trained signal rather than degenerate output,
though it's one clip, not a benchmark -- still treat detections as a lead
to review, not a confirmed incident.

Also not used: the model's pose branch. Box+class coordinates decode to
plausible pixel positions, but the 13 keypoints' visibility scores came
back near-zero (~0.000-0.001) on every real detection tested regardless
of how visible the person actually was -- the keypoint head looks
undertrained (the dataset's real signal is almost certainly the 2-class
action label, not precise pose), and Roboflow never published names for
this custom 13-point layout anyway. So this script only decodes
box+class, matching every other single-purpose detector in this project.

Detector I/O: standard Ultralytics YOLOv8s-pose export, nc=2, kpt_shape
[13,3] (kept in the graph, ignored here). Input images[1,3,640,640] RGB
letterboxed, /255. Output output0[1,45,8400] = 4 box (cx,cy,w,h, 640-space
pixels, already decoded) + 2 class scores (already sigmoid'd, same
convention verified empirically in every other YOLO detector here) +
39 keypoint values (unused). Needs transpose + confidence filter + NMS.

Like ContainerTracking, a single frame can't show whether shoplifting
behavior actually happened (it's an action, not a static pose) -- takes a
video (or ordered frame directory) for a real run, though a single image
is also accepted as a lighter sanity check (see --input docs). Reuses the
from-scratch SORT-style tracker (IOU + Hungarian) from ContainerTracking/
CustomerDwellTime to give each person a persistent ID and require the
shoplifting class to win N consecutive frames (--alert-frames) before
raising an alert, so single-frame classifier noise doesn't spam alerts.
"""

import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from scipy.optimize import linear_sum_assignment

CLASS_NAMES = ["normal", "shoplifting"]
ALERT_CLASS = "shoplifting"
DEFAULT_FPS_FALLBACK = 25.0

# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIRS = ("model", "models", "weights", "onnx", "checkpoints", ".")
# Nour190/shoplifting-yolo ships only best.pt on HF (no .onnx), so auto-fetch
# can't resolve it directly -- the exported model/shoplifting_yolov8s_pose.onnx
# is committed locally instead (see model/download.sh).
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
class ActionDetector:
    """Detects people and classifies each as normal/shoplifting in one pass
    (box + class share the same YOLO head; see module docstring for why the
    pose/keypoint part of the output is decoded nowhere in this file)."""

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

    def detect(self, img, confidence_threshold=0.5, nms_threshold=0.45):
        blob = self.preprocess(img)
        outputs = self.sess.run(self.output_names, {self.input_name: blob})
        preds = np.squeeze(outputs[0], 0).T  # [8400, 45]: cx,cy,w,h, 2 cls scores, 39 kpt (unused)
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
# (same approach as ContainerTracking/CustomerDwellTime).
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
        self.trail = [self._center()]

    def _center(self):
        x1, y1, x2, y2 = self.box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def update(self, detection):
        self.box = detection["box"]
        self.class_name = detection["class"]
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
class TheftRecord:
    """Per-track-ID bookkeeping so a single flickery frame can't raise an
    alert -- the shoplifting class has to win N consecutive frames
    (--alert-frames) first. Kept independent of the tracker's own track
    list so stats survive after a track is pruned."""

    def __init__(self):
        self.streak = 0
        self.shoplifting_frames = 0
        self.total_frames = 0
        self.alerted = False

    def update(self, class_name, alert_frames):
        self.total_frames += 1
        if class_name == ALERT_CLASS:
            self.shoplifting_frames += 1
            self.streak += 1
        else:
            self.streak = 0
        if self.streak >= alert_frames:
            self.alerted = True
        return self.streak >= alert_frames


# ---------------------------------------------------------------------------
def draw_frame(image, tracks, records, alert_frames):
    out = image.copy()
    h, w = out.shape[:2]
    scale = max(w, h) / 1400
    box_thick = max(2, int(3 * scale))
    label_scale = max(0.8, 1.1 * scale)
    label_thick = max(1, int(2 * scale))

    alert_count = 0
    for trk in tracks:
        if trk.misses > 0:
            continue
        rec = records[trk.id]
        is_alert = rec.streak >= alert_frames
        alert_count += int(is_alert)

        if is_alert:
            color = (0, 0, 255)
        elif trk.class_name == ALERT_CLASS:
            color = (0, 140, 255)  # suspicious this frame, not yet sustained
        else:
            color = (0, 200, 0)

        x1, y1, x2, y2 = map(int, trk.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thick)
        for i in range(1, len(trk.trail)):
            cv2.line(out, trk.trail[i - 1], trk.trail[i], color, max(1, box_thick // 2))

        label = f"ID {trk.id}  {'SHOPLIFTING' if is_alert else trk.class_name}  {trk.score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thick)
        label_y = max(th + baseline + 8, y1)
        cv2.rectangle(out, (x1, label_y - th - baseline - 8), (x1 + tw + 12, label_y), color, -1)
        cv2.putText(out, label, (x1 + 6, label_y - baseline - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, label_scale, (0, 0, 0), label_thick, cv2.LINE_AA)

    active = sum(1 for t in tracks if t.misses == 0)
    banner = f"PEOPLE: {active}   ALERTS: {alert_count}"
    banner_scale = max(1.0, 1.3 * scale)
    banner_thick = max(2, int(3 * scale))
    (tw, th), baseline = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, banner_scale, banner_thick)
    pad = int(12 * scale) + 8
    cv2.rectangle(out, (0, 0), (min(w, tw + pad * 2), th + baseline + pad * 2), (0, 0, 0), -1)
    color = (0, 0, 255) if alert_count > 0 else (0, 255, 0)
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
        description="Retail theft (shoplifting) detection sanity test (image, video, or frame directory in).")
    parser.add_argument("-i", "--input", required=True,
                        help="Path to a single image, a video file, OR a directory of ordered "
                             "frame images. A single image only shows the classifier's raw "
                             "per-frame call (no multi-frame alert debounce) -- pass a video for "
                             "a real run.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path: an image for single-image input, a video for video/"
                             "frame-dir input (or a directory if --frames is set). "
                             "Default: output.jpg / theft_tracked.mp4")
    parser.add_argument("-m", "--model", default=None,
                        help="Path to .onnx, a folder containing one, or a HF ref. "
                             "Omit to auto-discover in: " + ", ".join(DEFAULT_MODEL_DIRS))
    parser.add_argument("-c", "--conf-threshold", type=float, default=0.5)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--alert-frames", type=int, default=3,
                         help="Consecutive frames the 'shoplifting' class must win for the same "
                              "tracked person before an ALERT is raised (default: 3)")
    parser.add_argument("--iou-threshold", type=float, default=0.3,
                         help="Minimum IOU for the tracker to match a detection to an existing track")
    parser.add_argument("--max-age", type=int, default=15,
                         help="Frames a track survives without a matching detection")
    parser.add_argument("--fps", type=float, default=None,
                         help="Override the video's reported fps (informational only here)")
    parser.add_argument("--frames", action="store_true",
                         help="Write annotated frames to --output (a directory) instead of a video")
    args = parser.parse_args()

    single_image = Path(args.input).is_file() and Path(args.input).suffix.lower() in IMAGE_EXTS
    if single_image and args.frames:
        raise ValueError("--frames only applies to video/frame-directory input")
    if args.output is None:
        args.output = "output.jpg" if single_image else "theft_tracked.mp4"
    alert_frames = 1 if single_image else args.alert_frames

    detector = ActionDetector(resolve_model_path(args.model))
    tracker = PersonTracker(iou_threshold=args.iou_threshold, max_age=args.max_age)

    frames, detected_fps = load_frames(args.input)
    if not frames:
        raise FileNotFoundError(f"No frames found at {args.input}")
    fps = args.fps or detected_fps or DEFAULT_FPS_FALLBACK
    print(f"[input ] {len(frames)} frame(s) from {args.input} @ {fps:.1f} fps")
    if single_image:
        print("[note  ] single-image input -- showing the raw per-frame classification, no "
              "multi-frame alert debounce (run on a video for that)")

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
        detections = detector.detect(frame, confidence_threshold=args.conf_threshold,
                                      nms_threshold=args.nms_threshold)
        tracks = tracker.update(detections)

        active = [t for t in tracks if t.misses == 0]
        alerts_this_frame = []
        for trk in active:
            if trk.id not in records:
                records[trk.id] = TheftRecord()
            if records[trk.id].update(trk.class_name, alert_frames):
                alerts_this_frame.append(trk.id)

        print(f"  {name}: {len(active)} person(s) -> "
              f"{[f'ID{t.id}({t.class_name},{t.score:.2f})' for t in active]}"
              + (f"  ALERT: {alerts_this_frame}" if alerts_this_frame else ""))

        annotated = draw_frame(frame, tracks, records, alert_frames)
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

    alerted_ids = [tid for tid, rec in records.items() if rec.alerted]
    print(f"[result] {len(records)} distinct person(s) tracked across {len(frames)} frame(s), "
          f"{len(alerted_ids)} alerted: {sorted(alerted_ids)}")
    print(f"[done  ] wrote {args.output}")
