#!/bin/bash
# spacewalk01/yolov5-face-mask-detection: a custom YOLOv5s trained on the
# classic Kaggle "Face Mask Detection" dataset (with_mask/without_mask/
# mask_weared_incorrect). Ships a ready .pt directly in the GitHub repo
# (no HF repo). The checkpoint predates the unified `ultralytics` package
# and won't unpickle through it (`ModuleNotFoundError: No module named
# 'models'`), so it's exported with the classic yolov5 repo's own
# export.py rather than YOLO(...).export():

set -e
MODEL_DIR="$(dirname "$0")"
curl -sL "https://raw.githubusercontent.com/spacewalk01/yolov5-face-mask-detection/master/models/mask_yolov5.pt" -o "$MODEL_DIR/mask_yolov5.pt"

WORKDIR="$(mktemp -d)"
git clone --depth 1 https://github.com/ultralytics/yolov5.git "$WORKDIR/yolov5"
pip install -q -r "$WORKDIR/yolov5/requirements.txt"
python3 "$WORKDIR/yolov5/export.py" --weights "$MODEL_DIR/mask_yolov5.pt" --img 640 --include onnx --opset 12 --simplify
rm -rf "$WORKDIR"
