#!/usr/bin/env python3
"""
realsense_yolo_depth -- live RealSense depth + YOLOv8 detection, in Python.

The Python counterpart of ../../realsense_depth_customprocess_cpu (the
DeepStream/nvcustomprocess CPU pipeline), using the RealSense SDK directly
instead of GStreamer. For each frame it:

  1. grabs the aligned color + depth (Z16) frames from a RealSense camera,
  2. runs YOLOv8 (ONNX) object detection on the color image,
  3. samples the metric depth at each detection's centre and labels the box
     with "<class> <conf>% <distance> m",
  4. overlays the same metadata the GStreamer pipeline shows -- centre depth,
     device name/serial/USB/firmware, and the latest accel/gyro,
  5. shows the annotated color frame.

No depth colormap is drawn -- the depth is only used to compute the per-object
distance labels (and the centre-depth readout). The YOLOv8 detector class is
the same one from the ONNX-YOLOv8-Object-Detection repo, vendored into this
project's `yolov8/` package; only the depth fusion + overlay is new here.

============================================================================
EXPLICIT PIPELINE SPEC  (this file is the reference for the DeepStream build)
============================================================================
RealSense source (what the camera is asked to produce):
  color  : 640 x 480 @ 30 fps, format BGR8  (8-bit, 3 channels, interleaved)
  depth  : 640 x 480 @ 30 fps, format Z16   (16-bit; metres = pixel * depth_scale)
  imu    : accel + gyro streams (motion), enabled by default
  align  : depth is aligned INTO the color frame (rs.align(color)) -> after
           alignment the depth image has the COLOR resolution (640 x 480)
  depth_scale: read from the device at runtime (D455 ~= 0.001 m/unit)

YOLOv8 network (ONNX, run with onnxruntime):
  input  : 1 x 3 x 640 x 640, NCHW, float32              # tensor shape, explicit
  color  : RGB        (the BGR camera frame is converted BGR->RGB)
  resize : STRETCH to 640x640 (NOT letterbox; aspect not preserved)
  norm   : pixel / 255.0   (no mean/std)
  output : output0  1 x 84 x 8400  = 4 box (cx,cy,w,h) + 80 class scores
  decode : per-anchor argmax over the 80 class scores, threshold, multiclass NMS

Depth fusion:
  per box, distance = median of non-zero Z16 in a 7x7 patch at the box centre,
  times depth_scale -> metres. Boxes with no valid depth get no distance label.

DeepStream / nvinfer equivalents (for porting to a gst-launch pipeline):
  color BGR8 -> nvvideoconvert -> 'video/x-raw(memory:NVMM),format=NV12'
  YOLOv8 input 640x640, RGB, /255, STRETCH:
      net-scale-factor=0.0039215686   offsets=0;0;0
      model-color-format=0            infer-dims=3;640;640
      maintain-aspect-ratio=0         (stretch, to match this script)
      network-type=100  output-blob-names=output0  output-tensor-meta=1
  depth + imu + device-info arrive via customrealsensesrc attach-nvds-meta=true
  (see ../../realsense_yolo_depth_customprocess_cpu for the full pipeline).
or just ./run.sh
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from yolov8 import YOLOv8
from yolov8.utils import class_names, colors

HERE = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# EXPLICIT CONFIG -- every concrete pipeline parameter in one place.
# (argparse below defaults to these; override on the CLI when needed.)
# ============================================================================
# --- RealSense color stream ---
COLOR_WIDTH   = 640
COLOR_HEIGHT  = 480
COLOR_FPS     = 30
COLOR_FORMAT  = rs.format.bgr8        # 8-bit BGR, 3 interleaved channels

# --- RealSense depth stream ---
DEPTH_WIDTH   = 640
DEPTH_HEIGHT  = 480
DEPTH_FPS     = 30
DEPTH_FORMAT  = rs.format.z16         # 16-bit; metres = pixel * depth_scale

# --- alignment / IMU ---
ALIGN_TO      = rs.stream.color       # warp depth into the color frame
ENABLE_ACCEL  = True
ENABLE_GYRO   = True

# --- YOLOv8 network (ONNX) ---
YOLO_INPUT_W  = 640                   # network input width  (NCHW 1x3x640x640)
YOLO_INPUT_H  = 640                   # network input height
YOLO_COLOR    = "RGB"                 # model expects RGB (camera frame is BGR)
YOLO_NORM     = "x/255"               # normalization (no mean/std)
YOLO_RESIZE   = "stretch"             # NOT letterbox -- aspect not preserved
YOLO_OUTPUT   = "output0 [1,84,8400] = 4 box + 80 classes"
YOLO_CONF     = 0.5                   # detection confidence threshold
YOLO_IOU      = 0.5                   # NMS IoU threshold

# --- depth fusion ---
DEPTH_PATCH_HALF = 3                  # 7x7 patch (median of non-zero) per box


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="RealSense + YOLOv8 depth detection")
    p.add_argument("--model", default=os.path.join(HERE, "models", "yolov8n.onnx"),
                   help="path to the YOLOv8 ONNX model")
    p.add_argument("--width", type=int, default=COLOR_WIDTH, help="color/depth width")
    p.add_argument("--height", type=int, default=COLOR_HEIGHT, help="color/depth height")
    p.add_argument("--fps", type=int, default=COLOR_FPS)
    p.add_argument("--conf", type=float, default=YOLO_CONF, help="confidence threshold")
    p.add_argument("--iou", type=float, default=YOLO_IOU, help="NMS IoU threshold")
    p.add_argument("--input-w", type=int, default=YOLO_INPUT_W, help="YOLOv8 network input width")
    p.add_argument("--input-h", type=int, default=YOLO_INPUT_H, help="YOLOv8 network input height")
    p.add_argument("--max-depth", type=float, default=5.0,
                   help="centre-depth readout reference (metres)")
    p.add_argument("--no-imu", dest="imu", action="store_false", default=True,
                   help="disable the accel/gyro stream + overlay (on by default)")
    return p.parse_args()


# ---------------------------------------------------------------------------
def start_pipeline(args):
    """Start the RealSense pipeline; fall back gracefully if IMU is unavailable."""
    pipeline = rs.pipeline()

    def make_config(with_imu):
        cfg = rs.config()
        # color: BGR8, depth: Z16, both at the requested W x H @ fps
        cfg.enable_stream(rs.stream.color, args.width, args.height, COLOR_FORMAT, args.fps)
        cfg.enable_stream(rs.stream.depth, args.width, args.height, DEPTH_FORMAT, args.fps)
        if with_imu:
            if ENABLE_ACCEL:
                cfg.enable_stream(rs.stream.accel)
            if ENABLE_GYRO:
                cfg.enable_stream(rs.stream.gyro)
        return cfg

    try:
        profile = pipeline.start(make_config(args.imu))
        return pipeline, profile, args.imu
    except RuntimeError as e:
        if args.imu:
            print(f"[warn] could not start with IMU ({e}); retrying without IMU", file=sys.stderr)
            profile = pipeline.start(make_config(False))
            return pipeline, profile, False
        raise


def device_info_line(profile):
    """Build the 'name sn:.. usb:.. fw:..' line, like the GStreamer overlay."""
    dev = profile.get_device()

    def info(key):
        try:
            return dev.get_info(key)
        except Exception:
            return "?"

    name = info(rs.camera_info.name)
    serial = info(rs.camera_info.serial_number)
    fw = info(rs.camera_info.firmware_version)
    try:
        usb = dev.get_info(rs.camera_info.usb_type_descriptor)
    except Exception:
        usb = "?"
    return f"{name}  sn:{serial}  usb:{usb}  fw:{fw}"


def sample_depth_m(depth_img, cx, cy, depth_scale, half=DEPTH_PATCH_HALF):
    """Median of the non-zero depth in a (2*half+1)^2 patch around (cx, cy), in metres."""
    h, w = depth_img.shape[:2]
    x0, x1 = max(0, cx - half), min(w, cx + half + 1)
    y0, y1 = max(0, cy - half), min(h, cy + half + 1)
    patch = depth_img[y0:y1, x0:x1]
    nz = patch[patch > 0]
    if nz.size == 0:
        return 0.0
    return float(np.median(nz)) * depth_scale


def draw_label(img, text, x, y, color):
    """Filled caption box + white text with the top-left corner at (x, y)."""
    fs, th = 0.5, 1
    (tw, thh), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    yb = max(y, thh + 4)
    cv2.rectangle(img, (x, yb - thh - 4), (x + tw, yb), color, -1)
    cv2.putText(img, text, (x, yb - 2), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (255, 255, 255), th, cv2.LINE_AA)


def overlay_lines(img, lines, x=10, y0=22, lh=22):
    """Top-left white text block (centre depth / device / IMU)."""
    y = y0
    for s in lines:
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        y += lh


# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"model not found: {args.model}")

    print(f"loading YOLOv8: {args.model}")
    detector = YOLOv8(args.model, conf_thres=args.conf, iou_thres=args.iou)

    # yolov8n.onnx exports a DYNAMIC input shape (['batch', 3, 'height',
    # 'width']). Pin the network input to a concrete size EXPLICITLY (always),
    # so the resolution fed to the model is unambiguous (1 x 3 x H x W).
    detector.input_width = args.input_w
    detector.input_height = args.input_h

    pipeline, profile, imu_on = start_pipeline(args)
    align = rs.align(ALIGN_TO)                   # align depth into the color frame
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    dev_line = device_info_line(profile)

    # --- explicit config summary (so a glance tells you the whole pipeline) ---
    print("=" * 64)
    print("RealSense color : {}x{} @ {}fps  BGR8".format(args.width, args.height, args.fps))
    print("RealSense depth : {}x{} @ {}fps  Z16   depth_scale={:.6f} m/unit".format(
        args.width, args.height, args.fps, depth_scale))
    print("align           : depth -> color ({}x{} after alignment)".format(args.width, args.height))
    print("imu             : {}".format("accel+gyro ON" if imu_on else "OFF"))
    print("YOLOv8 input    : 1x3x{}x{} NCHW float32  | {} | norm {} | resize {}".format(
        args.input_h, args.input_w, YOLO_COLOR, YOLO_NORM, YOLO_RESIZE))
    print("YOLOv8 output   : {}".format(YOLO_OUTPUT))
    print("thresholds      : conf={}  iou={}".format(args.conf, args.iou))
    print("device          : {}".format(dev_line))
    print("=" * 64)

    win = "RealSense + YOLOv8 depth"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    t_prev, fps = time.time(), 0.0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color = aligned.get_color_frame()
            depth = aligned.get_depth_frame()
            if not color or not depth:
                continue

            color_img = np.asanyarray(color.get_data())          # HxWx3 BGR
            depth_img = np.asanyarray(depth.get_data())           # HxW uint16 (Z16)
            H, W = depth_img.shape[:2]

            # --- YOLOv8 detection on the color frame ---
            boxes, scores, class_ids = detector(color_img)

            # --- draw detections (+ per-object depth label) on the color frame ---
            for box, score, cid in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = box.astype(int)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                d = sample_depth_m(depth_img, cx, cy, depth_scale)
                col = tuple(int(c) for c in colors[cid])
                caption = f"{class_names[cid]} {int(score * 100)}%"
                if d > 0:
                    caption += f" {d:.2f}m"
                cv2.rectangle(color_img, (x1, y1), (x2, y2), col, 2)
                draw_label(color_img, caption, x1, y1, col)

            # --- metadata overlay (centre depth / device / IMU) ---
            cdist = sample_depth_m(depth_img, W // 2, H // 2, depth_scale, half=2)
            lines = [
                f"depth(center): {cdist:.3f} m   [{W}x{H}]   {fps:4.1f} FPS",
                dev_line,
            ]
            if imu_on:
                accel = frames.first_or_default(rs.stream.accel)
                gyro = frames.first_or_default(rs.stream.gyro)
                if accel:
                    a = accel.as_motion_frame().get_motion_data()
                    lines.append(f"accel: {a.x:+.2f} {a.y:+.2f} {a.z:+.2f}")
                if gyro:
                    g = gyro.as_motion_frame().get_motion_data()
                    lines.append(f"gyro:  {g.x:+.2f} {g.y:+.2f} {g.z:+.2f}")
            overlay_lines(color_img, lines)

            # --- show (annotated color frame only; no depth colormap) ---
            cv2.imshow(win, color_img)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):        # q or Esc
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
