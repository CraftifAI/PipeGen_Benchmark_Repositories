#!/usr/bin/env python3
"""smoke_test.py -- headless end-to-end check for the 2x2 perception dashboard.

Builds a few synthetic BGR frames, runs the FULL compute path (all four models),
exercises the EXACT composite() function application.py uses, asserts the output
shape/dtype, and writes the composited frames through a cv2.VideoWriter to verify
the --record path produces a non-empty file. Prints 'SMOKE OK' on success.

Run:  python3 smoke_test.py   (from this folder; no GUI needed)
"""

import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

ort.set_default_logger_severity(3)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import application as app
from application import (
    DET_MODEL, POSE_MODEL, SEG_MODEL, DEPTH_MODEL, COCO_TXT,
    PANEL_W, PANEL_H, composite,
    run_detection, run_segmentation, run_pose, run_depth,
)
from pipeline.detection import Detector
from pipeline.pose import PoseSession
from pipeline.segmentation import YOLOv11Seg
from pipeline.depth import MiDaSDepthEstimator


def main():
    rng = np.random.default_rng(0)
    # Two synthetic frames at different resolutions.
    frames = [
        rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
        rng.integers(0, 256, (640, 480, 3), dtype=np.uint8),
    ]

    print("Loading models ...")
    detector = Detector(DET_MODEL, conf_thres=0.5, iou_thres=0.5)
    seg = YOLOv11Seg(SEG_MODEL, COCO_TXT, bbox_threshold=0.6, nms_iou_threshold=0.4)
    pose_sess = PoseSession(POSE_MODEL)
    depth_est = MiDaSDepthEstimator(DEPTH_MODEL, input_size=256, use_gpu=False)

    out_w, out_h = PANEL_W * 2, PANEL_H * 2 + 28
    rec_path = os.path.join(HERE, "_smoke_out.mp4")
    writer = cv2.VideoWriter(rec_path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (out_w, out_h))
    assert writer.isOpened(), "VideoWriter failed to open"

    for i, frame in enumerate(frames):
        det_p = run_detection(detector, frame)
        seg_p = run_segmentation(seg, frame)
        pose_p = run_pose(pose_sess, frame)
        depth_p = run_depth(depth_est, frame)

        for name, p in (("det", det_p), ("seg", seg_p), ("pose", pose_p), ("depth", depth_p)):
            assert p.shape == (PANEL_H, PANEL_W, 3), f"{name} panel shape {p.shape}"
            assert p.dtype == np.uint8, f"{name} panel dtype {p.dtype}"

        hud = f"frame {i} | det {detector.infer_ms:.0f}ms seg {seg.infer_ms:.0f}ms " \
              f"pose {pose_sess.infer_ms:.0f}ms depth {depth_est.infer_ms:.0f}ms"
        canvas = composite(det_p, seg_p, pose_p, depth_p, hud)

        assert canvas.shape == (out_h, out_w, 3), f"canvas shape {canvas.shape}"
        assert canvas.dtype == np.uint8, f"canvas dtype {canvas.dtype}"
        writer.write(canvas)
        print(f"frame {i}: canvas {canvas.shape} | {hud}")

    writer.release()
    assert os.path.exists(rec_path), "record file not created"
    size = os.path.getsize(rec_path)
    assert size > 0, "record file is empty"
    print(f"recorded {rec_path} ({size} bytes)")
    os.remove(rec_path)

    print("SMOKE OK")


if __name__ == "__main__":
    main()
