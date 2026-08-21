#!/bin/bash
# Nour190/shoplifting-yolo (YOLOv8s-pose, 2 classes: normal/shoplifting).
# Ships only best.pt on Hugging Face, so it's exported locally rather than
# downloaded pre-converted:

pip install -q ultralytics huggingface_hub
python3 -c "
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
path = hf_hub_download('Nour190/shoplifting-yolo', 'best.pt')
YOLO(path).export(format='onnx', opset=12, simplify=True)
"
mv best.onnx "$(dirname "$0")/shoplifting_yolov8s_pose.onnx" 2>/dev/null || true
