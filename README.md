# PipeGen Benchmark Repository

This repository is a benchmark collection for PipeGen. It gathers multiple computer vision and perception projects that cover object detection, depth estimation, semantic segmentation, and multi-modal autonomous driving perception, making it a practical workspace for evaluating pipeline generation and orchestration across varied real-world AI repositories.

For model setup, each repository may include its own helper shell script such as `download.sh` in models folder, `download_models.sh`, or a similarly named downloader. Users can run the relevant script inside the target repository to fetch the required pretrained models and supporting assets before executing the pipeline.

## Repository Overview

`Bevfusion` is a bird's-eye-view perception repository focused on fusing camera and LiDAR information for autonomous driving workloads.  
It likely includes model setup, pretrained checkpoint download flows, and evaluation or demo utilities for 3D scene understanding tasks.  
As part of this benchmark set, it represents a large multi-modal project with heavier dependencies, richer model logic, and end-to-end perception pipelines.

`DeepLabV3-Plus-MobileNet-onnx` appears to center on semantic segmentation using a DeepLabV3+ architecture with a MobileNet backbone exported to ONNX.  
It is well suited for benchmarking lightweight vision inference pipelines where accuracy and deployment efficiency both matter.  
In this collection, it offers a focused example of image segmentation, ONNX model handling, and practical inference-oriented project structure.

`EfficientNet` looks like an image classification repository built around the EfficientNet ONNX model for compact and efficient inference.  
It likely provides a simple application entry point, model download flow, and label mapping for running classification on input images.  
Within the PipeGen benchmark, it contributes a lightweight classification workload that is useful for testing straightforward model loading and prediction pipelines.

`Facebook-RTDETR` appears to be an object detection repository based on the RT-DETR model family, likely adapted for ONNX inference and practical deployment.  
It seems suited for benchmarking transformer-style real-time detection pipelines that differ from YOLO-based architectures in both model design and runtime behavior.  
In this benchmark set, it adds another modern detection workload with pretrained model downloads, application-level inference, and visual output generation.

`Fast-SCNN` looks like a semantic segmentation repository centered on the Fast-SCNN architecture for efficient scene parsing.  
It likely targets lightweight segmentation use cases where real-time or edge-friendly inference is important, especially for road scenes or general pixel-wise labeling tasks.  
For PipeGen benchmarking, it contributes another segmentation-focused pipeline with a simpler application flow and an emphasis on speed-oriented deployment.

`Mask2Former-Instance-Segmentation` appears to be an instance segmentation repository built around the Mask2Former model for mask-level scene understanding.  
It likely supports label-aware segmentation of individual objects, making it useful for benchmarking richer outputs than plain bounding-box detection.  
Within the PipeGen benchmark suite, it adds a modern transformer-based segmentation workload with model downloads, application inference, and mask generation behavior.

`multi-model-perception-dashboard` is a self-contained four-model perception stack that runs YOLOv8 object detection, YOLO11 instance segmentation, 2D PAF pose estimation, and MiDaS monocular depth in parallel on a single webcam or video source, then composites the outputs into a labeled 2x2 dashboard with an FPS and per-model latency HUD.  
Inference is fanned out across a `ThreadPoolExecutor(4)` so all four ONNX models execute concurrently per frame, and the result can be displayed live, recorded to a video file, or both — with an included `smoke_test.py` that exercises the same composite path on synthetic frames headlessly.  
Within the PipeGen benchmark collection, it stands out as the most pipeline-orchestration-heavy entry: it stresses parallel multi-model scheduling, output fusion, and HUD compositing in a way that single-task repositories do not.

`ONNX-MiDaS-Depth-Estimation` is a depth estimation repository built around the MiDaS model running through ONNX Runtime for efficient inference.  
It seems intended for processing images, video, or webcam streams to generate monocular depth maps, which makes it a useful example of dense prediction beyond classification or detection.  
Within the PipeGen benchmark, it contributes a depth-focused workload with preprocessing, model execution, and visualization steps.

`ONNX-TopFormer-Semantic-Segmentation` looks like a semantic segmentation repository using the TopFormer architecture in ONNX format.  
It appears aimed at efficient dense scene parsing, likely with webcam or image inference paths that emphasize practical deployment and lightweight segmentation performance.  
For PipeGen benchmarking, it adds another segmentation pipeline with a distinct model family, ONNX runtime flow, and structured pixel-level prediction output.

`ONNX-Unidepth-Monocular-Metric-Depth-Estimation` appears to focus on metric monocular depth estimation using the UniDepth model in ONNX format.  
It likely goes beyond relative depth by targeting scale-aware predictions from a single image, making it especially relevant for applications that need more physically meaningful scene geometry.  
For PipeGen benchmarking, it adds another dense prediction workflow with modern ONNX deployment patterns and a different depth-estimation objective than MiDaS or stereo-based models.

`onnx-Ultra-Fast-Lane-Detection-Inference` is a lane detection repository using an ONNX version of the Ultra-Fast Lane Detection model.  
It appears aimed at detecting road lanes from driving imagery or video, making it especially relevant for real-time transportation and ADAS-style scenarios.  
For PipeGen benchmarking, it adds a specialized structured-vision task with temporal or road-scene inference patterns that differ from generic detection and segmentation projects.

`ONNX-YOLOv8-Object-Detection` is an object detection project that uses YOLOv8 models exported to ONNX for streamlined deployment.  
It likely provides scripts for loading models, running inference on images or video, and drawing detections in a simple application-ready format.  
For benchmarking PipeGen, it serves as a representative detection repository with common modern CV patterns, fast inference expectations, and deployment-friendly model packaging.

`Pose_Estimation` is a human pose estimation repository built around an ONNX model with keypoint and part-affinity-field decoding for multi-person skeletons.  
It supports running inference on images, videos, GIFs, or a live webcam stream, drawing limb connections and keypoints directly onto the output frames.  
Within the PipeGen benchmark collection, it adds a structured keypoint-prediction workload that differs from classification, detection, segmentation, and depth pipelines.

`realsense_yolo_depth` is a live Intel RealSense + YOLOv8 fusion pipeline that grabs aligned color and Z16 depth frames from the camera, runs YOLOv8 ONNX object detection on the color image, and labels each detection box with the median metric depth sampled at its centre.  
It also overlays device metadata (name, serial, USB, firmware) and live accelerometer/gyroscope readings, reusing the YOLOv8 ONNX detector class from `ONNX-YOLOv8-Object-Detection` so only the depth-fusion and overlay logic are new.  
Within the PipeGen benchmark collection, it contributes a sensor-driven multi-modal pipeline that couples a RealSense source with neural detection and metric depth fusion, making it a good probe for orchestrating real hardware streams alongside ONNX inference.

`ResNet-Image-Classification` looks like a straightforward image classification repository using a ResNet ONNX model and standard label mapping.  
It likely offers a compact inference entry point for classifying input images, making it a simple but useful benchmark for baseline computer vision pipelines.  
Within the PipeGen benchmark collection, it adds a classic classification workload that is easy to compare against lightweight or more modern architectures such as EfficientNet.

`ResNet-Object-Detection` appears to be an object detection repository using a ResNet-based detection model, likely tailored for traffic or scene-specific inference.  
It seems well suited for benchmarking practical detection pipelines that rely on classic backbone-driven models rather than only the newest detector families.  
For PipeGen benchmarking, it contributes another detection example with pretrained model setup, application inference, and labeled visual outputs in a deployment-oriented structure.

`yolo11n-instance-segmentation` appears to focus on instance segmentation using a compact YOLO11 segmentation model exported to ONNX.  
It likely supports running inference on images or video while producing both object-level detections and segmentation masks in a deployment-friendly workflow.  
Within the PipeGen benchmark suite, it adds a strong example of mask-based scene understanding that combines detection and segmentation in a single modern vision pipeline.

`yolov8-multi-camera-input` is an object detection repository that runs a YOLOv8 ONNX model across six synchronized camera streams (front, front-left, front-right, back, back-left, back-right) in a single batched inference pass.  
It is well suited for autonomous-driving-style surround-view perception, where multiple video inputs must be processed together and annotated with detections in real time.  
Within the PipeGen benchmark collection, it contributes a multi-input batched detection workload that stresses pipeline orchestration across parallel video sources rather than a single image or video stream.

`yolov8_pose_parallel` runs YOLOv8n object detection and YOLO11n-pose keypoint estimation concurrently on each frame via a `ThreadPoolExecutor(2)`, then overlays the detection boxes and the 17-keypoint COCO skeletons onto the same image with an FPS readout.  
The YOLOv8 wrapper reuses the `YOLOv8` class from `ONNX-YOLOv8-Object-Detection` (added to `sys.path`), while the pose head is a self-contained letterbox-and-decode port of the DeepStream `pose_render_cpu` reference, supporting webcam, video file, and headless `--output` recording paths.  
Within the PipeGen benchmark collection, it adds a focused two-model parallel pipeline that pairs detection with pose in a single per-frame fan-out, complementing the broader four-model `multi-model-perception-dashboard` with a smaller, more targeted concurrency example.
