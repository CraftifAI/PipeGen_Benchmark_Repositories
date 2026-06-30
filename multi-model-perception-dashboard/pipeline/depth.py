"""MiDaS monocular depth estimation (ONNX).

Copied from the source repo `ONNX-MiDaS-Depth-Estimation/video_depth_estimation.py`
(class MiDaSDepthEstimator + apply_colormap) so this folder is self-contained.

API:
    est = MiDaSDepthEstimator(model_path, input_size=256, use_gpu=False)
    depth_u8 = est.estimate_depth(frame)          # uint8 0..255 at original size
    color    = apply_colormap(depth_u8, cv2.COLORMAP_INFERNO)
"""

import time
from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort


class MiDaSDepthEstimator:
    """MiDaS Depth Estimation using ONNX Runtime."""

    def __init__(self, onnx_model_path: str, input_size: int = 256, use_gpu: bool = False):
        self.input_size = input_size
        providers = []
        if use_gpu:
            providers.append(('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
            }))
        providers.append('CPUExecutionProvider')

        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.infer_ms = 0.0

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_size, self.input_size))
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        img_normalized = (img_resized.astype(np.float32) / 255.0 - mean) / std
        input_tensor = img_normalized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        return input_tensor, original_size

    def postprocess(self, depth_output: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        depth = depth_output.squeeze()
        depth_resized = cv2.resize(depth, original_size)
        depth_min = depth_resized.min()
        depth_max = depth_resized.max()
        depth_normalized = (depth_resized - depth_min) / (depth_max - depth_min + 1e-8)
        depth_normalized = (depth_normalized * 255).astype(np.uint8)
        return depth_normalized

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        input_tensor, original_size = self.preprocess(image)
        t0 = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        self.infer_ms = (time.perf_counter() - t0) * 1000.0
        depth_output = outputs[0]
        return self.postprocess(depth_output, original_size)


def apply_colormap(depth_map: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    return cv2.applyColorMap(depth_map, colormap)
