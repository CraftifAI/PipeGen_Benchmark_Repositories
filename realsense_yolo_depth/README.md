# realsense_yolo_depth (Python)

Live **RealSense + YOLOv8 depth detection** in pure Python — the Python
counterpart of [`../../realsense_depth_customprocess_cpu`](../../realsense_depth_customprocess_cpu)
(the DeepStream/`nvcustomprocess` CPU pipeline), using the RealSense SDK
directly instead of GStreamer.

Per frame it:

1. grabs the **aligned color + depth (Z16)** frames from a RealSense camera,
2. runs **YOLOv8** (ONNX, via onnxruntime) object detection on the color image,
3. samples the metric depth at each detection's centre and **labels the box**
   with `<class> <conf>% <distance> m`,
4. overlays the same metadata the GStreamer pipeline shows — centre depth,
   device **name / serial / USB / firmware**, and the latest **accel / gyro**,
5. shows the annotated color frame.

> No depth colormap is drawn — the depth stream is used only to compute the
> per-object distance **labels** (and the centre-depth readout).

The YOLOv8 ONNX detector class is the same one from
`ONNX-YOLOv8-Object-Detection`, vendored into this project's `yolov8/` package
so the project is standalone; only the depth fusion + metadata overlay is new
here.

## Requirements

Needs a Python env with `pyrealsense2`, `onnxruntime`, `opencv-python`, and
`numpy` installed, plus a RealSense camera and a display.

## Run

```bash
./run.sh                 # 640x480 @30, YOLOv8n, conf 0.5
./run.sh --imu           # also overlay accel/gyro (D435i/D455/...)
./run.sh --conf 0.3      # lower the detection confidence threshold
./run.sh --max-depth 3   # centre-depth readout reference (metres)
```

`run.sh` just invokes `python3 realsense_yolo_depth.py "$@"` (override the
interpreter with `PY=...`). Press **q** or **Esc** to quit. All flags:
`python3 realsense_yolo_depth.py --help`.

## Files

```
realsense_yolo_depth/
├── realsense_yolo_depth.py     the pipeline (RealSense + YOLOv8 + depth labels)
├── yolov8/                     vendored YOLOv8 ONNX inference package
├── models/
│   └── yolov8n.onnx            the YOLOv8n ONNX model
├── run.sh
└── README.md
```

## How the depth label works

`rs.align(rs.stream.color)` warps the depth frame into the color camera's
geometry, so a detection box in the color image indexes the matching depth
pixels directly. For each box we take the **median of the non-zero Z16 values**
in a small patch around the box centre (robust to depth holes) and multiply by
the sensor's `depth_scale` to get metres. Boxes with no valid depth (`z==0`
everywhere in the patch) are labelled without a distance.

## Notes

- **Explicit config:** every concrete pipeline parameter (color/depth
  resolution + format + fps, alignment, IMU, YOLOv8 input shape, color order,
  normalization, resize mode, output tensor) is declared in the `EXPLICIT
  CONFIG` block at the top of `realsense_yolo_depth.py` and printed as a summary
  at startup — it doubles as the spec for the DeepStream port.
- **Dynamic ONNX input:** `yolov8n.onnx` exports a dynamic input shape
  (`['batch', 3, 'height', 'width']`), so the script pins the network input
  explicitly to `640x640` (override with `--input-w` / `--input-h`).
- **IMU is optional:** `--imu` enables the accel/gyro streams; if the device or
  permissions don't support them, the pipeline logs a warning and continues
  without IMU rather than failing.
- **Model path:** defaults to `models/yolov8n.onnx`. Point at any YOLOv8 ONNX
  with `--model /path/to/yolov8X.onnx`.
