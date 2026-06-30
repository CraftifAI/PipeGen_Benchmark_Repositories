# yolov8_pose_parallel (Python)

**YOLOv8 object detection + YOLO11-pose estimation, run in parallel per frame.**

- **Detection:** YOLOv8n (COCO 80 classes) — `yolov8n.onnx`.
- **Pose:** YOLO11n-pose (COCO 17-keypoint skeleton) — `yolo11n-pose.onnx`.

Both ONNX models run concurrently on each frame (separate threads — onnxruntime
releases the GIL during `session.run`), then the detection boxes and the pose
skeletons are drawn on the same frame (boxes first, skeletons on top).

## Run

```bash
./run.sh                                  # default webcam (/dev/video0)
./run.sh --input people.mp4 --output out.mp4 --no-show
```

Press **q**/**Esc** to quit. `--help` lists flags (`--conf`, `--pose-conf`,
model paths).

## Files

```
yolov8_pose_parallel/
├── detectors.py     YoloV8Detector + PoseEstimator
├── yolov8_pose.py   the parallel pipeline
├── yolov8/          vendored YOLOv8 ONNX inference package
├── models/          yolov8n.onnx, yolo11n-pose.onnx
├── run.sh
└── README.md
```

`detectors.py` uses the vendored `yolov8` package (the same `YOLOv8` class from
`ONNX-YOLOv8-Object-Detection`, copied in so the project is standalone); the
pose parse is ported from the DeepStream `pose_render_cpu.cpp` (threshold +
letterbox-undo + greedy NMS, COCO skeleton).

## Notes

- Both models export a dynamic ONNX input; the wrappers pin the network input to
  640×640.
- CPU `onnxruntime` works; a CUDA execution provider is used automatically if
  present.
- Verified that both sessions load + run in parallel and draw; for visible
  skeletons run it on a webcam or a clip with people (`--input people.mp4`).
