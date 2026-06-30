#!/usr/bin/env python3
"""application.py -- Four-model perception dashboard (2x2 multi-view).

Runs the FULL perception stack on a single webcam and composites every model's
output into ONE synchronized 2x2 grid:

    +-----------------------+-----------------------+
    | YOLOv8 DETECTION      | YOLO11 SEGMENTATION   |
    +-----------------------+-----------------------+
    | 2D POSE (PAF)         | MiDaS DEPTH (inferno) |
    +-----------------------+-----------------------+

All four models run IN PARALLEL on each frame via a ThreadPoolExecutor(4)
(ONNX Runtime releases the GIL during inference). An FPS + per-model latency HUD
is drawn on top. Output can be shown live, suppressed (headless), and/or
recorded to a video file so the result is verifiable without a display.

Usage
-----
    python3 application.py                       # webcam 0
    python3 application.py --source 1            # webcam 1
    python3 application.py --source clip.mp4     # a video file
    python3 application.py --no-display          # headless (use with --record)
    python3 application.py --record out.mp4      # write composited frames to a file
    python3 application.py --no-display --record out.mp4

Single webcam: just run with the default --source 0; all four panels are derived
from that one stream. Press 'q' or ESC in the display window to quit.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import onnxruntime as ort

ort.set_default_logger_severity(3)        # hush graph-optimization warnings

from pipeline.detection import Detector, draw_detections
from pipeline.pose import PoseSession, preprocess as pose_preprocess, parse as pose_parse, draw as pose_draw
from pipeline.segmentation import YOLOv11Seg
from pipeline.depth import MiDaSDepthEstimator, apply_colormap

HERE = os.path.dirname(os.path.abspath(__file__))
DET_MODEL = os.path.join(HERE, "models", "yolov8m.onnx")
POSE_MODEL = os.path.join(HERE, "models", "pose_estimation.onnx")
SEG_MODEL = os.path.join(HERE, "models", "yolo11n-seg.onnx")
DEPTH_MODEL = os.path.join(HERE, "models", "midas_v21_small.onnx")
COCO_TXT = os.path.join(HERE, "coco.txt")

PANEL_W, PANEL_H = 480, 360               # per-panel size (modest for speed)


# ---------- per-model run functions (each returns a BGR panel-sized image) ----------
def run_detection(detector, frame):
    """YOLOv8 detection -> boxes drawn on the frame."""
    tensor = detector.preprocess([frame])
    raws = detector.infer(tensor)
    boxes, scores, class_ids = detector.postprocess(raws, [frame])[0]
    out = draw_detections(frame, boxes, scores, class_ids)
    return _fit_panel(out)


def run_segmentation(seg, frame):
    """YOLO11 instance segmentation -> 640x640 annotated image."""
    annotated = seg.segment(frame)        # 640x640 BGR with masks drawn
    return _fit_panel(annotated)


def run_pose(pose_sess, frame):
    """2D PAF-based pose -> skeletons drawn on the frame."""
    x = pose_preprocess(frame)
    cmap, paf = pose_sess(x)
    objects, peaks = pose_parse(cmap, paf)
    out = frame.copy()
    pose_draw(out, objects, peaks)
    return _fit_panel(out)


def run_depth(depth_est, frame):
    """MiDaS monocular depth -> inferno-colored depth map."""
    depth_u8 = depth_est.estimate_depth(frame)
    color = apply_colormap(depth_u8, cv2.COLORMAP_INFERNO)
    return _fit_panel(color)


# ---------- compositing ----------
def _fit_panel(img):
    """Resize any BGR image to a single (PANEL_H, PANEL_W) panel."""
    if img.shape[1] != PANEL_W or img.shape[0] != PANEL_H:
        img = cv2.resize(img, (PANEL_W, PANEL_H))
    return img


def _label(panel, text):
    """Draw a title banner on a panel (returns the same array)."""
    cv2.rectangle(panel, (0, 0), (PANEL_W, 24), (0, 0, 0), -1)
    cv2.putText(panel, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0), 1, cv2.LINE_AA)
    return panel


def composite(det_panel, seg_panel, pose_panel, depth_panel, hud=""):
    """Build the labeled 2x2 dashboard canvas (+ optional HUD bar).

    Returns a BGR uint8 image of shape (2*PANEL_H + 28, 2*PANEL_W, 3).
    This is the EXACT function used by both application.py and smoke_test.py.
    """
    tl = _label(_fit_panel(det_panel.copy()), "YOLOv8 DETECTION")
    tr = _label(_fit_panel(seg_panel.copy()), "YOLO11 SEGMENTATION")
    bl = _label(_fit_panel(pose_panel.copy()), "2D POSE (PAF)")
    br = _label(_fit_panel(depth_panel.copy()), "MiDaS DEPTH")

    top = np.hstack([tl, tr])
    bottom = np.hstack([bl, br])
    grid = np.vstack([top, bottom])

    bar = np.zeros((28, grid.shape[1], 3), dtype=np.uint8)
    if hud:
        cv2.putText(bar, hud, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, grid])


def open_source(src):
    cap = cv2.VideoCapture(int(src)) if str(src).isdigit() else cv2.VideoCapture(src)
    return cap if cap.isOpened() else None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0", help="Webcam index or video path (default 0).")
    p.add_argument("--no-display", action="store_true",
                   help="Run headless (no cv2 window). Pair with --record.")
    p.add_argument("--record", default=None,
                   help="Write the composited dashboard to this video file.")
    p.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold.")
    p.add_argument("--iou", type=float, default=0.5, help="Detection NMS IoU threshold.")
    p.add_argument("--seg-conf", type=float, default=0.6, help="Segmentation bbox threshold.")
    p.add_argument("--seg-iou", type=float, default=0.4, help="Segmentation NMS IoU threshold.")
    p.add_argument("--fps", type=int, default=15, help="Declared FPS for --record output.")
    return p.parse_args()


def main():
    args = parse_args()

    for label, path in (("detection", DET_MODEL), ("pose", POSE_MODEL),
                        ("segmentation", SEG_MODEL), ("depth", DEPTH_MODEL),
                        ("coco labels", COCO_TXT)):
        if not os.path.exists(path):
            sys.exit(f"{label} file not found: {path}")

    cap = open_source(args.source)
    if cap is None:
        sys.exit(f"Could not open source: {args.source}")

    print("Loading YOLOv8m detector ...")
    detector = Detector(DET_MODEL, conf_thres=args.conf, iou_thres=args.iou)
    print("Loading YOLO11 segmenter ...")
    seg = YOLOv11Seg(SEG_MODEL, COCO_TXT,
                     bbox_threshold=args.seg_conf, nms_iou_threshold=args.seg_iou)
    print("Loading 2D pose model ...")
    pose_sess = PoseSession(POSE_MODEL)
    print("Loading MiDaS depth model ...")
    depth_est = MiDaSDepthEstimator(DEPTH_MODEL, input_size=256, use_gpu=False)
    print("ONNX providers:", detector.session.get_providers())

    writer = None
    out_w, out_h = PANEL_W * 2, PANEL_H * 2 + 28
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record, fourcc, args.fps, (out_w, out_h))
        print(f"Recording dashboard -> {args.record} ({out_w}x{out_h}@{args.fps})")

    if not args.no_display:
        cv2.namedWindow("Perception Dashboard (2x2)", cv2.WINDOW_NORMAL)

    pool = ThreadPoolExecutor(max_workers=4)
    print("Running. Press 'q' or ESC to quit.")
    n, t0 = 0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Source ended / read failed; stopping.", file=sys.stderr)
                break

            # All four models run IN PARALLEL on the same frame.
            f_det = pool.submit(run_detection, detector, frame)
            f_seg = pool.submit(run_segmentation, seg, frame)
            f_pose = pool.submit(run_pose, pose_sess, frame)
            f_depth = pool.submit(run_depth, depth_est, frame)
            det_p = f_det.result()
            seg_p = f_seg.result()
            pose_p = f_pose.result()
            depth_p = f_depth.result()

            n += 1
            fps = n / max(time.perf_counter() - t0, 1e-6)
            hud = (f"{fps:4.1f} FPS | det {detector.infer_ms:4.0f}ms | "
                   f"seg {seg.infer_ms:4.0f}ms | pose {pose_sess.infer_ms:4.0f}ms | "
                   f"depth {depth_est.infer_ms:4.0f}ms")
            canvas = composite(det_p, seg_p, pose_p, depth_p, hud)

            if writer is not None:
                writer.write(canvas)
            if not args.no_display:
                cv2.imshow("Perception Dashboard (2x2)", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        pool.shutdown(wait=False)
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
