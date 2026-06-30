import os
import cv2
import argparse
import numpy as np
import onnxruntime as ort


def classify(model_path, labels_path):
    # Load ONNX model
    session = ort.InferenceSession(model_path)

    # Get input/output names dynamically
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Load labels
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    # Open webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_disp = frame.copy()

        # 🔹 Preprocess (HWC → CHW, normalize)
        frame_in = cv2.resize(frame, (224, 224))
        frame_in = frame_in.astype(np.float32) / 255.0
        frame_in = np.transpose(frame_in, (2, 0, 1))  # HWC → CHW
        frame_in = np.expand_dims(frame_in, axis=0)   # NCHW

        # 🔹 Inference
        output = session.run([output_name], {input_name: frame_in})[0]

        # 🔹 Postprocess
        pred_id = int(np.argmax(output[0]))
        label = labels[pred_id] if pred_id < len(labels) else str(pred_id)

        # 🔹 Display
        cv2.putText(frame_disp, f"{label}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)

        cv2.imshow("output", frame_disp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_path",
        default=os.path.join(dir_path,"models", "efficientnetb2_batch1.onnx"),
        help="Path to ONNX model"
    )
    parser.add_argument(
        "-l", "--labels_path",
        default=os.path.join(dir_path, "labels.txt"),
        help="Path to labels file"
    )

    args = parser.parse_args()
    classify(args.model_path, args.labels_path)
