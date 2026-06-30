import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import argparse
import os

class RTDETR:
    def __init__(self, model_path):
        self.ort_session = ort.InferenceSession(model_path)
        input_info = self.ort_session.get_inputs()[0]
        self.input_channels, self.input_height, self.input_width = input_info.shape[1:]
        class_names_str = self.ort_session.get_modelmeta().custom_metadata_map['names']
        self.class_names = eval(class_names_str)

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.input_width, self.input_height))
        image = np.array(image).astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image

    def run_inference(self, image_path):
        input_image = self.preprocess_image(image_path)
        input_name = self.ort_session.get_inputs()[0].name
        outputs = self.ort_session.run(None, {input_name: input_image})
        return outputs

    def post_process(self, outputs, confidence_threshold=0.3):
        bboxes = outputs[0][:, :4]  # First 4 values: bounding box [cx, cy, width, height]
        logits = outputs[0][:, 4:]  # Next values: class scores
        best_class_indices = np.argmax(logits, axis=-1)
        best_class_confidences = np.max(logits, axis=-1)
        high_confidence_indices = best_class_confidences > confidence_threshold
        best_class_indices = best_class_indices[high_confidence_indices]
        best_class_confidences = best_class_confidences[high_confidence_indices]
        bboxes = bboxes[high_confidence_indices]
        cx, cy, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        x_min = (cx - 0.5 * w)
        y_min = (cy - 0.5 * h)
        x_max = (cx + 0.5 * w)
        y_max = (cy + 0.5 * h)
        detections = []
        for i in range(len(best_class_indices)):
            detection = {
                "class": int(best_class_indices[i]),
                "confidence": float(best_class_confidences[i]),
                "bbox": [float(x_min[i]), float(y_min[i]), float(x_max[i]), float(y_max[i])]
            }
            detections.append(detection)

        return detections

    def draw_detections(self, image_path, detections, output_path):
        image = cv2.imread(image_path)
        img_height, img_width, _ = image.shape
        for det in detections:
            class_idx = det["class"]
            class_name = self.class_names[class_idx]
            confidence = det["confidence"]
            bbox = det["bbox"]
            x_min, y_min, x_max, y_max = bbox
            x_min = int(x_min * img_width)
            y_min = int(y_min * img_height)
            x_max = int(x_max * img_width)
            y_max = int(y_max * img_height)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imwrite(output_path, image)
        cv2.imshow("output", image)
        input_image = cv2.imread(image_path)
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
    parser = argparse.ArgumentParser(description="Run object detection using an ONNX model.")
    parser.add_argument("-m", "--model", default=os.path.join(dir, "models", "rtdetr-l.onnx"), help="Path to the ONNX model file.")
    parser.add_argument("-i", "--image", default=os.path.join(dir, "..", "images", "img1.jpg"), help="Path to the input image file.")
    parser.add_argument("-o", "--output", default=os.path.join(dir, "output.jpg"), help="Path to save the output image.")
    args = parser.parse_args()
    model = RTDETR(args.model)
    output = model.run_inference(args.image)
    detections = model.post_process(output[0])
    model.draw_detections(args.image, detections, args.output)
