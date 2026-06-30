#!/usr/bin/env bash
# Run the RealSense + YOLOv8 depth-detection pipeline.
# Requires a Python env with pyrealsense2 / onnxruntime / opencv installed.
#
#   ./run.sh                 640x480 @30, YOLOv8n, conf 0.5, IMU overlay ON
#   ./run.sh --no-imu        disable the accel/gyro overlay
#   ./run.sh --conf 0.3      lower the detection confidence threshold
#   ./run.sh --max-depth 3   centre-depth readout reference (metres)
#
# Any flags are forwarded to realsense_yolo_depth.py (see --help).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY="${PY:-python3}"

exec "$PY" "$HERE/realsense_yolo_depth.py" "$@"
