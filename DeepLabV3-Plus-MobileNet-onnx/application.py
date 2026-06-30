import os
import cv2
import argparse
import numpy as np
import onnxruntime as ort

def generate_colormap(num_classes: int) -> np.ndarray:
    colormap = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        hue = int(180 * i / num_classes)
        col = np.uint8([[[hue, 255, 255]]])
        rgb = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)[0, 0]
        colormap[i] = rgb
    return colormap

def apply_color_map(class_map: np.ndarray) -> np.ndarray:
    num_classes = np.max(class_map) + 1
    colormap = generate_colormap(num_classes)
    height, width = class_map.shape
    color_map_output = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(1, num_classes):
        color_map_output[class_map == i] = colormap[i]
    return color_map_output

def overlay_segmentation(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    color_mask = apply_color_map(mask)
    return cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)

def segmentation(input_video, output_video, model_path: str):
    session = ort.InferenceSession(model_path)
    model_input_width, model_input_height = 520, 520
    frame_width, frame_height = 800, 600
    for inp in session.get_inputs():
        print("Input name:", inp.name, "shape:", inp.shape)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 20.0, (frame_width, frame_height))
    cap = cv2.VideoCapture(input_video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (model_input_width, model_input_height))
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        frame_input = np.expand_dims(frame_normalized, axis=0) # (1, H, W, C)
        frame_input = np.transpose(frame_input, (0, 3, 1, 2))  # (1, C, H, W)

        output_mask = session.run(None, {"image": frame_input})[0][0]

        frame = cv2.resize(frame, (frame_width, frame_height))
        mask_resized = cv2.resize(
            output_mask.astype(np.uint8),
            (frame_width, frame_height),
            interpolation=cv2.INTER_NEAREST
        )
        blended_output = overlay_segmentation(frame, mask_resized, alpha=0.4)

        cv2.imshow("Segmentation", blended_output)
        out.write(blended_output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_video", default=os.path.join(dir, "..","videos","video2.mp4"),
                                                                       help="Path to input video file")
    parser.add_argument("-o", "--output_video", default=os.path.join(dir,"output.mp4"), 
                                                                        help="Path to output video file")
    parser.add_argument("-m", "--model_path",
                        default=os.path.join(dir, "models", "DeepLabV3-Plus-MobileNet.onnx"))
    args = parser.parse_args()
    segmentation(args.input_video, args.output_video, args.model_path)
