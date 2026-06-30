#!/usr/bin/env python3
"""YOLOv8 object detection + YOLO11-pose estimation, run IN PARALLEL per frame.

Two ONNX models run concurrently on each frame (separate threads), then the
detection boxes and the pose skeletons are drawn on the same frame.

    ./run.sh                                  # default webcam
    python yolov8_pose.py --input people.mp4 --output out.mp4
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

from detectors import YoloV8Detector, PoseEstimator, open_source

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 + YOLO11-pose in parallel")
    p.add_argument("--input", default="0", help="video path or webcam index")
    p.add_argument("--yolo-model", default=os.path.join(HERE, "models", "yolov8n.onnx"))
    p.add_argument("--pose-model", default=os.path.join(HERE, "models", "yolo11n-pose.onnx"))
    p.add_argument("--conf", type=float, default=0.4, help="YOLOv8 confidence")
    p.add_argument("--pose-conf", type=float, default=0.25, help="pose person confidence")
    p.add_argument("--output", default=None, help="write annotated video here")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print("pipeline: yolov8 detection + yolo11-pose estimation (parallel)")
    print(f"  model 1 (yolov8): {args.yolo_model}")
    print(f"  model 2 (pose)  : {args.pose_model}")
    yolo = YoloV8Detector(args.yolo_model, conf=args.conf)
    pose = PoseEstimator(args.pose_model, conf=args.pose_conf)

    cap = open_source(args.input)
    if not cap.isOpened():
        sys.exit(f"could not open input: {args.input}")

    writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    win = "YOLOv8 + Pose (parallel)"
    if not args.no_show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    pool = ThreadPoolExecutor(max_workers=2)
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t0 = time.time()
            f_yolo = pool.submit(yolo.infer, frame)
            f_pose = pool.submit(pose.infer, frame)
            yolo_res, pose_res = f_yolo.result(), f_pose.result()

            yolo.draw(frame, yolo_res)     # boxes first
            pose.draw(frame, pose_res)     # skeletons on top

            fps = 0.9 * fps + 0.1 * (1.0 / max(time.time() - t0, 1e-6))
            cv2.putText(frame, f"{fps:4.1f} FPS",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            if writer:
                writer.write(frame)
            if not args.no_show:
                cv2.imshow(win, frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    finally:
        pool.shutdown()
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
