#!/bin/bash
# Same Ultralytics YOLO11s (COCO-pretrained) person detector reused by
# VehicleDetection/DockUtilization/QueueLengthMonitoring/CustomerDwellTime.
# It ships only as .pt on Hugging Face, so it's exported locally rather
# than downloaded pre-converted:

pip install -q ultralytics
python3 -c "from ultralytics import YOLO; YOLO('yolo11s.pt').export(format='onnx', opset=12, simplify=True)"
# The export lands in the CWD; move it next to this script only when that
# is a different directory -- `mv f ./f` is an error ("are the same file")
# and made the documented `cd model && bash download.sh` exit non-zero.
SRC="$PWD/yolo11s.onnx"
DEST="$(cd "$(dirname "$0")" && pwd)/yolo11s.onnx"
[ "$SRC" = "$DEST" ] || mv "$SRC" "$DEST"
