# PipeGen Benchmark Repository

This repository is a benchmark collection for PipeGen. It gathers multiple computer vision and perception projects that cover object detection, depth estimation, semantic segmentation, and multi-modal autonomous driving perception, alongside a set of applied retail, logistics, surveillance, and workplace-safety use cases, making it a practical workspace for evaluating pipeline generation and orchestration across varied real-world AI repositories.

For model setup, each repository may include its own helper shell script such as `download.sh` in models folder, `download_models.sh`, or a similarly named downloader. Users can run the relevant script inside the target repository to fetch the required pretrained models and supporting assets before executing the pipeline. Model weights are not committed here, so running that downloader is a required first step for most folders.

## Repository Overview

`AbandonedObjectDetection` flags unattended luggage in surveillance footage by tracking COCO backpack/handbag/suitcase detections from an Ultralytics YOLO11s ONNX detector with a SORT-style IOU + Hungarian tracker.  
An object is reported as abandoned only when it stays stationary (displacement under 20% of its own box diagonal) and no person's box is within three box-diagonals of it, continuously, for `--abandoned-seconds`.  
Within the PipeGen benchmark collection, it contributes a security-analytics workload that layers multi-object tracking and owner-association geometry on top of a plain detector rather than a purpose-trained class.

`ANPR` is an automatic number plate recognition pipeline that chains two ONNX models: a YOLOv9-t 384x384 license-plate detector with NMS baked into the graph, then a Compact Convolutional Transformer OCR head that reads 10 character slots over a 37-symbol alphabet plus a 66-way region prediction.  
It is a from-scratch onnxruntime implementation rather than a wrapper around the `fast_alpr` package, and `model/download.sh` fetches both weights along with the OCR alphabet/region config the recognizer cannot run without.  
For PipeGen benchmarking, it adds a genuinely two-stage detect-then-recognize workload where the second model consumes crops produced by the first.

`Bevfusion` is a bird's-eye-view perception repository focused on fusing camera and LiDAR information for autonomous driving workloads.  
It likely includes model setup, pretrained checkpoint download flows, and evaluation or demo utilities for 3D scene understanding tasks.  
As part of this benchmark set, it represents a large multi-modal project with heavier dependencies, richer model logic, and end-to-end perception pipelines.

`CheckoutMonitoring` reports per-lane checkout status from a single image or a sequence of frames, using the same Ultralytics YOLO11s COCO person detector reused across the retail use cases in this collection.  
Each lane in `--zone-file` carries its own cashier polygon and queue polygon, and a lane resolves to OPEN - SERVING, OPEN - IDLE, CLOSED, or UNSTAFFED - QUEUE! — the last being the state a store manager actually needs surfaced.  
Within the PipeGen benchmark, it contributes a multi-zone rule engine over detection output, and it expects a low `--conf-threshold` because ceiling-mounted wide-angle shots make people small.

`ContainerFillLevel` estimates how full a container is from one monocular image by running Depth-Anything-V2-metric through ONNX Runtime and reading the per-row median depth profile of the frame.  
The largest smoothed discontinuity in that profile is taken as the fill boundary — nearer side filled, farther side empty — and fill percentage is the filled row fraction; a frame with no qualifying jump reads as effectively full.  
For PipeGen benchmarking, it adds a dense metric-depth workload whose output is a derived scalar rather than boxes or masks, and it is a heuristic rather than a calibrated volumetric measurement.

`ConveyorBeltMonitoring` runs a YOLOv8 detector fine-tuned for conveyor lines (`abhiejit/package-conveyor-ggu-yolo2`) over four classes: bag, box, carton, and conveyor.  
Input is `images[1,3,640,640]` RGB letterboxed to 640 with grey padding and scaled by 1/255; output `output0[1,8,8400]` is raw 4-box plus 4-class and needs the usual transpose, confidence filter, and NMS.  
Within the PipeGen benchmark collection, it contributes a logistics-domain detection workload driven entirely by the contract embedded in the ONNX metadata.

`CustomerDwellTime` measures how long each individual shopper stays inside a zone, combining the YOLO11s person detector, a from-scratch SORT-style tracker (IOU + Hungarian matching), and a polygon zone from `--zone-file`.  
Dwell is the cumulative seconds a track's foot-point spends inside the zone, converted from frame counts using the source fps, so it needs a video or an ordered directory of frames rather than a single image.  
For PipeGen benchmarking, it adds a detect-track-and-clock pipeline where the answer only exists across frames, with no cross-session re-identification once a track is lost.

`CustomerHeatmaps` builds a retail foot-traffic heatmap by accumulating a small Gaussian blob at each detected person's foot-point into a running 2D density map over every frame of a video.  
The accumulator is normalized, colorized with the JET colormap, and alpha-blended over a representative frame, so the output is a single composited image rather than per-frame annotations.  
Within the PipeGen benchmark collection, it contributes a spatial-accumulation workload that reduces an entire video to one aggregate visualization.

`DeepLabV3-Plus-MobileNet-onnx` appears to center on semantic segmentation using a DeepLabV3+ architecture with a MobileNet backbone exported to ONNX.  
It is well suited for benchmarking lightweight vision inference pipelines where accuracy and deployment efficiency both matter.  
In this collection, it offers a focused example of image segmentation, ONNX model handling, and practical inference-oriented project structure.

`EfficientNet` looks like an image classification repository built around the EfficientNet ONNX model for compact and efficient inference.  
It likely provides a simple application entry point, model download flow, and label mapping for running classification on input images.  
Within the PipeGen benchmark, it contributes a lightweight classification workload that is useful for testing straightforward model loading and prediction pipelines.

`EmptyShelfDetection` locates out-of-stock gaps on retail shelves with `wiwiewei18/smart-shelf-tracker`, a single-class YOLO11n whose confidence is badly miscalibrated by its 497-image training set.  
It therefore runs at a deliberately very low confidence threshold and suppresses the resulting noise with two filters: drop boxes centred in the bottom `floor_frac` of the frame, and drop boxes below `min_area_frac` of the image area.  
For PipeGen benchmarking, it is a useful example of a detector whose usable operating point sits far from the conventional 0.25-0.5 default.

`Facebook-RTDETR` appears to be an object detection repository based on the RT-DETR model family, likely adapted for ONNX inference and practical deployment.  
It seems suited for benchmarking transformer-style real-time detection pipelines that differ from YOLO-based architectures in both model design and runtime behavior.  
In this benchmark set, it adds another modern detection workload with pretrained model downloads, application-level inference, and visual output generation.

`FaceMaskDetection` classifies faces as with_mask, without_mask, or mask_weared_incorrect using `spacewalk01/yolov5-face-mask-detection`, a YOLOv5s trained on the classic 853-image Kaggle mask dataset.  
It is the one repository here on the classic YOLOv5 head convention — `output0[1,25200,8]` with a separate objectness column, so final confidence is objectness times class score rather than class score alone.  
The checkpoint predates the unified `ultralytics` package and will not unpickle through it, so `model/download.sh` exports it with the original yolov5 repo's own `export.py`.

`Fast-SCNN` looks like a semantic segmentation repository centered on the Fast-SCNN architecture for efficient scene parsing.  
It likely targets lightweight segmentation use cases where real-time or edge-friendly inference is important, especially for road scenes or general pixel-wise labeling tasks.  
For PipeGen benchmarking, it contributes another segmentation-focused pipeline with a simpler application flow and an emphasis on speed-oriented deployment.

`LoiteringDetection` alerts on continuous presence rather than presence itself, pairing the YOLO11s person detector with a SORT-style tracker and the named polygon zones of `RestrictedAreaMonitoring`.  
Each track carries a session clock over consecutive in-zone frames that resets the moment it leaves, so someone who walks away and comes back starts a fresh visit instead of continuing the old one; `--loiter-seconds` defaults to 30.  
Within the PipeGen benchmark collection, it sits between the harmless-zone dwell metric and the always-alert restricted zone, and it needs a video for a real time-threshold reading.

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

`onnx-Ultra-Fast-Lane-Detection-Inference` is a lane detection repository using an ONNX version of the Ultra-Fast Lane Detection model.  
It appears aimed at detecting road lanes from driving imagery or video, making it especially relevant for real-time transportation and ADAS-style scenarios.  
For PipeGen benchmarking, it adds a specialized structured-vision task with temporal or road-scene inference patterns that differ from generic detection and segmentation projects.

`ONNX-Unidepth-Monocular-Metric-Depth-Estimation` appears to focus on metric monocular depth estimation using the UniDepth model in ONNX format.  
It likely goes beyond relative depth by targeting scale-aware predictions from a single image, making it especially relevant for applications that need more physically meaningful scene geometry.  
For PipeGen benchmarking, it adds another dense prediction workflow with modern ONNX deployment patterns and a different depth-estimation objective than MiDaS or stereo-based models.

`ONNX-YOLOv8-Object-Detection` is an object detection project that uses YOLOv8 models exported to ONNX for streamlined deployment.  
It likely provides scripts for loading models, running inference on images or video, and drawing detections in a simple application-ready format.  
For benchmarking PipeGen, it serves as a representative detection repository with common modern CV patterns, fast inference expectations, and deployment-friendly model packaging.

`Pose_Estimation` is a human pose estimation repository built around an ONNX model with keypoint and part-affinity-field decoding for multi-person skeletons.  
It supports running inference on images, videos, GIFs, or a live webcam stream, drawing limb connections and keypoints directly onto the output frames.  
Within the PipeGen benchmark collection, it adds a structured keypoint-prediction workload that differs from classification, detection, segmentation, and depth pipelines.

`ppe-kit` detects personal protective equipment with `Tanishjain9/yolov8n-ppe-detection-6classes`, a YOLOv8n covering gloves, vest, goggles, helmet, mask, and safety_shoe at roughly 0.81 mAP@50.  
Input is `images[1,3,640,640]` RGB letterboxed and scaled by 1/255; output `output0[1,10,8400]` is raw 4-box plus 6-class and needs transpose, confidence filter, and NMS.  
For PipeGen benchmarking, it contributes a compact edge-oriented safety-compliance detector, and its own README carries the full per-class metrics and dataset breakdown.

`QueueLengthMonitoring` counts how many people stand inside a configurable queue polygon, filtering the YOLO11s COCO detector down to the person class.  
Queue length here is the in-zone headcount that feeds an estimated-wait-time calculation; it does not infer queue order or fit a physical line shape, so a serpentine stanchion layout has to be described through `--zone-file`.  
Within the PipeGen benchmark collection, it is the simplest of the zone-rule retail workloads and a useful baseline against the dwell-time and checkout variants.

`realsense_yolo_depth` is a live Intel RealSense + YOLOv8 fusion pipeline that grabs aligned color and Z16 depth frames from the camera, runs YOLOv8 ONNX object detection on the color image, and labels each detection box with the median metric depth sampled at its centre.  
It also overlays device metadata (name, serial, USB, firmware) and live accelerometer/gyroscope readings, reusing the YOLOv8 ONNX detector class from `ONNX-YOLOv8-Object-Detection` so only the depth-fusion and overlay logic are new.  
Within the PipeGen benchmark collection, it contributes a sensor-driven multi-modal pipeline that couples a RealSense source with neural detection and metric depth fusion, making it a good probe for orchestrating real hardware streams alongside ONNX inference.

`ResNet-Image-Classification` looks like a straightforward image classification repository using a ResNet ONNX model and standard label mapping.  
It likely offers a compact inference entry point for classifying input images, making it a simple but useful benchmark for baseline computer vision pipelines.  
Within the PipeGen benchmark collection, it adds a classic classification workload that is easy to compare against lightweight or more modern architectures such as EfficientNet.

`ResNet-Object-Detection` appears to be an object detection repository using a ResNet-based detection model, likely tailored for traffic or scene-specific inference.  
It seems well suited for benchmarking practical detection pipelines that rely on classic backbone-driven models rather than only the newest detector families.  
For PipeGen benchmarking, it contributes another detection example with pretrained model setup, application inference, and labeled visual outputs in a deployment-oriented structure.

`RestrictedAreaMonitoring` treats any presence inside a marked danger zone as a security event, running the YOLO11s person detector, a SORT-style tracker, and a polygon membership test on every frame.  
Each track resolves to CLEAR, IN ZONE, or INTRUSION! — the last after `--alert-frames` consecutive in-zone frames (default 2, enough to absorb a missed detection at the zone boundary) and reported with the seconds spent inside.  
Omitting `--zone-file` makes the whole frame restricted, which suits a camera whose entire field of view is off-limits, such as a locked server room.

`ShelfOccupancy` reports a continuous shelf fill percentage from `foduucom/product-detection-in-shelf-yolov8`, a two-class detector over empty and product.  
Occupancy is the product box area divided by the combined product-plus-empty area, so it measures fill only across the shelf regions the model actually classified, not a segmentation of the whole surface.  
The model was trained on retail-cooler imagery and performs best on fridge and cooler shelves; its image loader is EXIF-aware because phone photos are commonly stored sideways.

`SmokeDetection` runs `rabahdev/fire-smoke-yolov8n`, a YOLOv8n fine-tuned on the real-photograph D-Fire dataset over two classes, smoke and fire, with both surfaced since either is safety-critical.  
A SORT-style tracker gives each detected region a persistent-ish ID so that `--alert-frames` consecutive detections are required before an alert fires, guarding against a single-frame false positive on a diffusing plume.  
Within the PipeGen benchmark collection, it contributes an environmental-hazard detection workload validated against both positive and clean-negative real imagery.

`TheftDetection` uses `Nour190/shoplifting-yolo`, a YOLOv8s-Pose model with two classes, normal and shoplifting, tracked across frames with the same SORT-style IOU + Hungarian tracker used elsewhere in this collection.  
An alert requires `--alert-frames` consecutive shoplifting classifications on the same track, so per-frame classifier flicker does not fire it.  
For PipeGen benchmarking, it adds a pose-backbone behaviour classifier whose verdict is only meaningful as a run of frames rather than as a single detection.

`UnsafeBehaviorDetection` detects smoking with `Enos-123/smoking-detection`, a single-class YOLO11m trained to find cigarettes.  
Input is `images[1,3,640,640]` RGB letterboxed and scaled by 1/255; output `output0[1,5,8400]` is raw 4-box plus one class score and needs transpose, confidence filter, and NMS — every surviving detection is by definition the unsafe behaviour.  
Within the PipeGen benchmark collection, it contributes a small-object detection workload where the target occupies very few pixels relative to the person holding it.

`VehicleDetection` runs Ultralytics' official COCO-pretrained YOLO11s through ONNX Runtime and counts the vehicle classes — bicycle, car, motorcycle, bus, train, truck, and boat.  
Everything else in COCO-80 is still detected and drawn as a completeness check, but only the vehicle classes feed the count banner.  
For PipeGen benchmarking, it is the plain-detection baseline of this use-case group, sharing its `output0[1,84,8400]` decode contract with most of the retail and surveillance entries.

`WorkerFallDetection` runs a YOLO11s-Pose ONNX export in the marcoslucianops DeepStream-Yolo-Pose layout, where the head output is transposed in-graph to `[1,8400,56]` — 4 box, 1 confidence, and 17 COCO keypoints of (x, y, visibility).  
Box decode, objectness sigmoid, and keypoint pixel-space decode are all baked in by the Ultralytics Pose head at export time, so only single-class NMS is left for the application to do.  
Falls are estimated from body orientation — a wide, short box plus a near-horizontal shoulder-to-hip line — which is a deliberately approximate single-frame heuristic, not a trained fall classifier.

`yolo11n-instance-segmentation` appears to focus on instance segmentation using a compact YOLO11 segmentation model exported to ONNX.  
It likely supports running inference on images or video while producing both object-level detections and segmentation masks in a deployment-friendly workflow.  
Within the PipeGen benchmark suite, it adds a strong example of mask-based scene understanding that combines detection and segmentation in a single modern vision pipeline.

`yolov8-multi-camera-input` is an object detection repository that runs a YOLOv8 ONNX model across six synchronized camera streams (front, front-left, front-right, back, back-left, back-right) in a single batched inference pass.  
It is well suited for autonomous-driving-style surround-view perception, where multiple video inputs must be processed together and annotated with detections in real time.  
Within the PipeGen benchmark collection, it contributes a multi-input batched detection workload that stresses pipeline orchestration across parallel video sources rather than a single image or video stream.

`yolov8_pose_parallel` runs YOLOv8n object detection and YOLO11n-pose keypoint estimation concurrently on each frame via a `ThreadPoolExecutor(2)`, then overlays the detection boxes and the 17-keypoint COCO skeletons onto the same image with an FPS readout.  
The YOLOv8 wrapper reuses the `YOLOv8` class from `ONNX-YOLOv8-Object-Detection` (added to `sys.path`), while the pose head is a self-contained letterbox-and-decode port of the DeepStream `pose_render_cpu` reference, supporting webcam, video file, and headless `--output` recording paths.  
Within the PipeGen benchmark collection, it adds a focused two-model parallel pipeline that pairs detection with pose in a single per-frame fan-out, complementing the broader four-model `multi-model-perception-dashboard` with a smaller, more targeted concurrency example.
