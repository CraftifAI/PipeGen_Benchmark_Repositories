"""Multi-frame sanity test for customer foot-traffic heatmaps.

Like tracking/counting, a heatmap is fundamentally a multi-frame concept
-- it visualizes WHERE people spent time across a video, which a single
image can't show at all. Takes a video (or a directory of ordered frame
images) instead of one image.

No dedicated "customer heatmap" model exists (this is a standard retail-
analytics technique built from person detection + spatial accumulation,
not a trained visual class). Reuses the already-verified Ultralytics
YOLO11s (COCO-pretrained) person detector from VehicleDetection/
DockUtilization/QueueLengthMonitoring.

Method: for every frame, detect people and accumulate a small Gaussian
"heat" blob at each person's foot-point (bottom-center of their box --
approximates where they're standing, not the center of their whole body)
into a running 2D density accumulator sized to the frame. After all
frames, the accumulator is normalized, colorized (JET colormap, the
standard convention for foot-traffic heatmaps), and alpha-blended over a
representative frame from the video.

Detector I/O: identical to the sibling use cases (input images[1,3,640,640]
RGB letterboxed, /255; output [1,84,8400] raw 4-box+80-class COCO, needs
transpose + confidence filter + NMS).
"""

import argparse
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
def foot_point(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, y2)


def splat_gaussian(accumulator, center, radius):
    """Adds a small Gaussian blob at `center` into `accumulator` -- much
    cheaper than blurring the whole frame-sized accumulator every frame,
    since only a radius x radius neighborhood is touched per detection."""
    h, w = accumulator.shape
    cx, cy = int(center[0]), int(center[1])
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (radius / 2) ** 2))
    accumulator[y0:y1, x0:x1] += blob


def load_frames(source):
    p = Path(source)
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


def _text_with_backing(img, text, org, font_scale, thickness, fg, bg=(0, 0, 0), pad=8):
    """Draws text on a solid backing rectangle so it stays readable over any
    part of the photo -- important for a non-technical audience who won't
    squint to find low-contrast text."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, fg, thickness, cv2.LINE_AA)
    return tw, th + baseline


def draw_legend(img, x, y, bar_w, bar_h, scale):
    """A labelled Low-to-High color gradient strip, so the JET colors mean
    something to someone who has never seen a heatmap before.

    The bar is widened to at least fit its own two labels (and the whole group
    nudged back inside the frame), because the label font scales with the frame
    while the requested bar width does not -- on a 4K frame the labels are ~550px
    each against a 320px bar, so they overlapped each other and ran off the edge.

    Returns the y of the top of the label text so the caller can stack above it.
    """
    img_h, img_w = img.shape[:2]
    font_scale = max(0.6, 0.8 * scale)
    thick = max(1, int(2 * scale))
    label1, label2 = "Low foot traffic", "High foot traffic"
    (tw1, th1), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
    (tw2, th2), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
    bar_w = int(min(max(bar_w, tw1 + tw2 + int(40 * scale)), img_w - 48))
    x = int(min(max(24, x), img_w - bar_w - 24))
    # JET maps 0 -> blue and 255 -> red, so the ramp must ASCEND left to right to
    # agree with the "Low" (left) / "High" (right) labels printed above it.
    gradient = np.linspace(0, 255, bar_w).astype(np.uint8).reshape(1, -1)
    gradient = np.repeat(gradient, bar_h, axis=0)
    gradient_colored = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
    img[y:y + bar_h, x:x + bar_w] = gradient_colored
    cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), (255, 255, 255), max(1, int(2 * scale)))
    label_y = y - 12
    _text_with_backing(img, label1, (x, label_y), font_scale, thick, (255, 255, 255))
    _text_with_backing(img, label2, (x + bar_w - tw2, label_y), font_scale, thick, (255, 255, 255))
    return label_y - max(th1, th2) - 8


def draw_heatmap(background, accumulator, num_frames, total_detections, fps=None, alpha=0.55):
    h, w = accumulator.shape
    scale = max(w, h) / 1400

    norm = accumulator / (accumulator.max() + 1e-9)
    norm_gamma = np.power(norm, 0.5)  # gamma boost so lower-traffic areas stay visible
    heat_u8 = (norm_gamma * 255).astype(np.uint8)
    heat_u8 = cv2.GaussianBlur(heat_u8, (0, 0), sigmaX=3)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    mask = (norm_gamma > 0.03)[..., None]
    out = background.copy()
    blended = cv2.addWeighted(colored, alpha, background, 1 - alpha, 0)
    out = np.where(mask, blended, out).astype(np.uint8)

    # Point out the single busiest spot explicitly -- don't make a non-technical
    # viewer infer it from color alone.
    peak_y, peak_x = np.unravel_index(np.argmax(accumulator), accumulator.shape)
    if accumulator.max() > 0:
        marker_r = max(8, int(14 * scale))
        cv2.circle(out, (peak_x, peak_y), marker_r, (255, 255, 255), max(2, int(3 * scale)))
        cv2.circle(out, (peak_x, peak_y), max(3, int(4 * scale)), (255, 255, 255), -1)
        label = "BUSIEST SPOT"
        label_scale = max(0.9, 1.1 * scale)
        label_thick = max(2, int(3 * scale))
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thick)
        lx = min(max(0, peak_x - tw // 2), w - tw - 16)
        # `ly` is the text BASELINE; the backing box drawn by _text_with_backing
        # extends a further (baseline + pad) below it, so that has to be part of
        # the clearance or the box paints over the marker it is pointing at.
        ly = max(th + 10, peak_y - marker_r - 14 - baseline - 10)
        cv2.line(out, (peak_x, peak_y - marker_r), (peak_x, ly + 6), (255, 255, 255), max(1, int(2 * scale)))
        _text_with_backing(out, label, (lx, ly), label_scale, label_thick, (255, 255, 0), pad=10)

    # Title banner, in plain language a non-technical viewer can read at a glance.
    title = "CUSTOMER FOOT TRAFFIC HEATMAP"
    title_scale = max(1.3, 1.7 * scale)
    title_thick = max(3, int(4 * scale))
    pad = int(14 * scale) + 10
    (tw, th), baseline = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_thick)
    banner_h = th + baseline + pad * 2
    cv2.rectangle(out, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.putText(out, title, (pad, th + pad), cv2.FONT_HERSHEY_SIMPLEX, title_scale, (255, 255, 255), title_thick, cv2.LINE_AA)

    # Plain-language summary line -- explicitly NOT phrased as a unique
    # customer count, since this accumulates detections across every frame
    # (the same person walking through is counted many times, once per
    # frame they appear in) -- overstating it as "customers" would mislead
    # a reader who takes the number at face value.
    if fps and fps > 0:
        duration_s = num_frames / fps
        coverage = f"~{duration_s:.0f} seconds of video ({num_frames} frames analyzed)"
    else:
        coverage = f"{num_frames} frames analyzed"
    summary = f"{coverage}  |  avg {total_detections / max(1, num_frames):.1f} people visible per frame"
    sub_scale = max(0.9, 1.15 * scale)
    sub_thick = max(2, int(2 * scale))
    (_, sub_th), sub_baseline = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, sub_scale, sub_thick)
    cv2.rectangle(out, (0, banner_h), (w, banner_h + sub_th + sub_baseline + 20), (0, 0, 0), -1)
    cv2.putText(out, summary, (pad, banner_h + sub_th + 10), cv2.FONT_HERSHEY_SIMPLEX,
                sub_scale, (0, 255, 0), sub_thick, cv2.LINE_AA)

    legend_w, legend_h = int(min(320, w * 0.25)), max(18, int(22 * scale))
    legend_top = draw_legend(out, w - legend_w - 24, h - legend_h - 24, legend_w, legend_h, scale)

    caption = "Colors show where people spent the most time on camera, not a live customer count."
    cap_scale = max(0.6, 0.8 * scale)
    cap_thick = max(1, int(2 * scale))
    # Stack the caption ABOVE the legend labels. At the old fixed y it shared a
    # baseline with them, so the two black text backings painted over each other.
    (_, _), cap_baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, cap_scale, cap_thick)
    _text_with_backing(out, caption, (24, legend_top - cap_baseline - 12), cap_scale, cap_thick,
                       (255, 255, 255))

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer foot-traffic heatmap sanity test (video or frame directory in).")
    parser.add_argument("-i", "--input", required=True, help="Video file OR a directory of ordered frame images")
    parser.add_argument("-o", "--output", default="heatmap.jpg")
    parser.add_argument("-m", "--model", default="model/yolo11s.onnx")
    parser.add_argument("-c", "--conf-threshold", type=float, default=0.35)
    parser.add_argument("--radius", type=int, default=None,
                         help="Gaussian blob radius in pixels (default: ~4%% of frame width)")
    parser.add_argument("--background-frame", type=int, default=0,
                         help="Index of the frame to use as the heatmap's background image")
    parser.add_argument("--overlay-boxes", action="store_true",
                         help="Also save a debug video of per-frame detections to verify the detector itself")
    parser.add_argument("--boxes-output", default="detections.mp4")
    args = parser.parse_args()

    detector = PersonDetector(args.model)
    frames, fps = load_frames(args.input)
    if not frames:
        raise FileNotFoundError(f"No frames found at {args.input}")
    print(f"[input ] {len(frames)} frame(s) from {args.input}" + (f" @ {fps:.1f} fps" if fps else ""))

    h, w = frames[0][1].shape[:2]
    radius = args.radius or max(10, int(0.04 * w))
    accumulator = np.zeros((h, w), dtype=np.float32)

    writer = None
    if args.overlay_boxes:
        writer = cv2.VideoWriter(args.boxes_output, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))

    total_detections = 0
    for name, frame in frames:
        people = [d for d in detector.detect(frame, confidence_threshold=args.conf_threshold)
                  if d["class"] == PERSON_CLASS]
        total_detections += len(people)
        for p in people:
            splat_gaussian(accumulator, foot_point(p["box"]), radius)
        print(f"  {name}: {len(people)} person(s)")
        if writer is not None:
            dbg = frame.copy()
            for p in people:
                x1, y1, x2, y2 = map(int, p["box"])
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
            writer.write(dbg)

    if writer is not None:
        writer.release()
        print(f"[done  ] wrote debug detections video: {args.boxes_output}")

    background = frames[min(args.background_frame, len(frames) - 1)][1]
    result = draw_heatmap(background, accumulator, len(frames), total_detections, fps=fps)
    cv2.imwrite(args.output, result)

    print(f"[result] {total_detections} person-detections accumulated across {len(frames)} frame(s), "
          f"peak cell density={accumulator.max():.1f}")
    print(f"[done  ] wrote {args.output}")
