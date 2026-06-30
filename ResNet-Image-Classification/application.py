import cv2
import numpy as np
import onnxruntime as ort


def load_labels(label_path):
    with open(label_path, "r") as f:
        return [line.strip() for line in f.readlines()]


def preprocess(image):
    # Resize
    image = cv2.resize(image, (224, 224))

    # BGR → RGB (IMPORTANT for most models)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize
    image = image.astype(np.float32) / 255.0

    # HWC → CHW
    image = np.transpose(image, (2, 0, 1))

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image


def classify(model_path, label_path):
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print("Input:", input_name)
    print("Output:", output_name)

    labels = load_labels(label_path)

    # 🔥 More reliable webcam open (Linux fix)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    # Optional: set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame not received")
            break

        input_tensor = preprocess(frame)

        output = session.run(
            [output_name],
            {input_name: input_tensor}
        )[0]

        probs = output[0]

        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])

        label = labels[class_id] if class_id < len(labels) else str(class_id)

        # Draw result
        text = f"{label}: {confidence:.2f}"
        cv2.putText(frame, text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.imshow("Classification", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    classify(
        "./models/resnet50_b1.onnx",
        "./labels.txt"
    )
