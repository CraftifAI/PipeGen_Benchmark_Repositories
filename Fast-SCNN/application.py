import os
import cv2
import numpy as np
import onnxruntime as ort
import argparse

COLORS = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
    [0, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
])

def preprocess_image(image, input_shape):
    resized_image = cv2.resize(image, (input_shape[3], input_shape[2]))  # Resize to model input
    normalized_image = resized_image.astype(np.float32) / 255.0          # Normalize to [0, 1]
    input_tensor = np.transpose(normalized_image, (2, 0, 1))            # HWC to CHW
    input_tensor = np.expand_dims(input_tensor, axis=0)                 # Add batch dimension
    return input_tensor

def postprocess_output(output_tensor):
    pred = np.argmax(output_tensor.squeeze(0), axis=0)  # Shape: [height, width]
    color_mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(COLORS):
        color_mask[pred == class_id] = color
    return color_mask

def run_inference(model_path, image_path, output_path):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_image = cv2.imread(image_path)
    if input_image is None:
        print("Error: Unable to read image.")
        return
    input_tensor = preprocess_image(input_image, input_shape)
    outputs = session.run(None, {input_name: input_tensor})
    output_tensor = outputs[0]
    color_mask = postprocess_output(output_tensor)
    cv2.imwrite(output_path, color_mask)
    cv2.imshow("output", color_mask)
    cv2.imshow("input", input_image)
    while True:
        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' to exit
            break
        if cv2.getWindowProperty("output", cv2.WND_PROP_VISIBLE) < 1 or cv2.getWindowProperty("input", cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Run image segmentation using an ONNX model.")
    parser.add_argument("-m", "--model", default=os.path.join(dir, "models", "fast_scnn.onnx"), help="Path to the ONNX model file.")
    parser.add_argument("-i", "--image", default=os.path.join(dir, "..", "images", "img1.jpg"), 
                                                              help="Path to the input image file.")
    parser.add_argument("-o", "--output", default=os.path.join(dir, "output.jpg"), help="Path to save the output image.")
    args = parser.parse_args()
    run_inference(args.model, args.image, args.output)
