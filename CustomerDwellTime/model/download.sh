#!/bin/bash
# Same Ultralytics YOLO11s (COCO-pretrained) person detector reused by
# VehicleDetection/DockUtilization/QueueLengthMonitoring/CustomerHeatmaps.
# It ships only as .pt on Hugging Face, so it's exported locally rather
# than downloaded pre-converted:

pip install -q ultralytics
python3 -c "from ultralytics import YOLO; YOLO('yolo11s.pt').export(format='onnx', opset=12, simplify=True)"
mv yolo11s.onnx "$(dirname "$0")/yolo11s.onnx"
