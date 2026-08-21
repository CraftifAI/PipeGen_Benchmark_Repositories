#!/bin/bash
# rabahdev/fire-smoke-yolov8n (YOLOv8n fine-tuned on D-Fire, 2 classes:
# smoke/fire). Ships only best.pt on Hugging Face, so it's exported
# locally rather than downloaded pre-converted:
set -euo pipefail

# Ultralytics writes the .onnx next to the SOURCE .pt (i.e. inside the
# Hugging Face cache), not into the current directory -- so copy from the
# path export() actually returns instead of guessing ./best.onnx.
DEST="$(cd "$(dirname "$0")" && pwd)/fire_smoke_yolov8n.onnx"

pip install -q ultralytics huggingface_hub

DEST="$DEST" python3 -c "
import os, shutil
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
path = hf_hub_download('rabahdev/fire-smoke-yolov8n', 'best.pt')
out = YOLO(path).export(format='onnx', opset=12, simplify=True)
dest = os.environ['DEST']
shutil.copyfile(out, dest)
print('[download] exported %s -> %s' % (out, dest))
"
