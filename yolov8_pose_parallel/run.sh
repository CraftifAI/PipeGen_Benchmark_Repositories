#!/usr/bin/env bash
# YOLOv8 detection + YOLO11-pose estimation in parallel (Python/onnxruntime).
#   ./run.sh                       default webcam (/dev/video0)
#   ./run.sh --input people.mp4 --output out.mp4
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"
exec "$PY" "$HERE/yolov8_pose.py" "$@"
