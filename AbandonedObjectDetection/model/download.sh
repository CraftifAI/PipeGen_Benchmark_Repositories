#!/bin/bash
# Same Ultralytics YOLO11s (COCO-pretrained) detector reused by
# VehicleDetection/DockUtilization/CustomerDwellTime/RestrictedAreaMonitoring/
# LoiteringDetection -- luggage classes (backpack/handbag/suitcase) and
# "person" are both already in its COCO-80 label set, so no custom-trained
# abandoned-object model is needed (see application.py docstring). Ships
# only .pt on Hugging Face, so it's exported locally rather than
# downloaded pre-converted:

pip install -q ultralytics
python3 -c "from ultralytics import YOLO; YOLO('yolo11s.pt').export(format='onnx', opset=12, simplify=True)"
# The export lands in the CWD. Only move it if that is not already the
# script's own directory -- when this is run as `cd model && bash download.sh`
# source and destination are the same file and plain `mv` aborts with
# "are the same file" (exit 1).
DEST="$(dirname "$0")/yolo11s.onnx"
if [ ! yolo11s.onnx -ef "$DEST" ]; then
    mv yolo11s.onnx "$DEST"
fi
