# Perception All-4 Dashboard (2x2)

Runs the full perception stack on one webcam — YOLOv8 detection, YOLO11 instance
segmentation, 2D PAF pose, and MiDaS monocular depth — all in parallel and
composited into one synchronized, labeled 2x2 dashboard with an FPS/latency HUD.

## Run

```bash
pip install -r requirements.txt

# Live, default webcam (single webcam is all you need — every panel comes from it):
python3 application.py

# Pick a different camera or a video file:
python3 application.py --source 1
python3 application.py --source clip.mp4

# Headless (no display) + record the composited dashboard to a file:
python3 application.py --no-display --record out.mp4

# Just record while also displaying:
python3 application.py --record out.mp4
```

Press `q` or `ESC` in the window to quit. Useful flags: `--conf`/`--iou`
(detection), `--seg-conf`/`--seg-iou` (segmentation), `--fps` (record framerate).

## Headless smoke test

```bash
python3 smoke_test.py   # builds synthetic frames, runs all 4 models + the
                        # exact composite, checks the VideoWriter path; prints SMOKE OK
```

## Layout

- `application.py` — entry point: parallel inference (ThreadPoolExecutor(4)) + 2x2 composite.
- `pipeline/detection.py` — batched YOLOv8 detector + draw_detections.
- `pipeline/segmentation.py` — YOLO11 instance segmentation.
- `pipeline/pose.py` — 2D PAF pose session + decode + draw.
- `pipeline/depth.py` — MiDaS depth estimator + colormap.
- `models/` — bundled ONNX models (self-contained; CPU-only ORT).
- `coco.txt` — COCO class labels used by the segmentation panel.

This folder is fully self-contained: all models and code are copied in, with no
dependencies back to sibling repositories.
