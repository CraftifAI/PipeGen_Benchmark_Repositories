#!/bin/bash
# Same Ultralytics YOLO11s (COCO-pretrained) person detector reused by
# VehicleDetection/DockUtilization/QueueLengthMonitoring/CustomerDwellTime.
# It ships only as .pt on Hugging Face, so it's exported locally rather
# than downloaded pre-converted:

# Run from the directory holding this script so the export lands here no
# matter where the caller invoked it from. The previous `mv yolo11s.onnx
# "$(dirname "$0")/yolo11s.onnx"` aborted with
#   mv: 'yolo11s.onnx' and './yolo11s.onnx' are the same file   (exit 1)
# for the natural `cd model && bash download.sh` invocation.
cd "$(dirname "$0")" || exit 1

pip install -q ultralytics
python3 -c "from ultralytics import YOLO; YOLO('yolo11s.pt').export(format='onnx', opset=12, simplify=True)"
