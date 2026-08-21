---
license: mit
tags:
- yolo
- yolov8
- object-detection
- ppe
- safety-detection
- onnx
- jetson
- edge-ai
---

# 🦺 YOLOv8n PPE Detection (6 Classes)

A lightweight and fast Personal Protective Equipment (PPE) detection model trained on 6 safety classes, optimized for edge devices such as **NVIDIA Jetson Orin**, **ONNX Runtime**, and **TensorRT** deployment.

---

# 1. 📌 Model Details

- **Model name:** YOLOv8n PPE Detection – 6 Classes  
- **Architecture:** Ultralytics YOLOv8n (Nano)  
- **Task:** Object Detection (PPE / Safety Gear)  
- **Framework:** PyTorch (Ultralytics), exported to ONNX  
- **Input size:** `1 × 3 × 640 × 640`  
- **Output shape (ONNX):** `1 × 10 × 8400`  
  - 4 channels → bounding box (x, y, w, h)  
  - 6 channels → class logits  
- **Number of classes:** `6`  

---

# 2. 🏷️ Classes & Dataset Distribution

### **Class List with Image Counts**

| ID | Class Name   | Image Count |
|----|--------------|-------------|
| 0  | Gloves       | 2,693       |
| 1  | Vest         | 4,418       |
| 2  | goggles      | 1,431       |
| 3  | helmet       | 2,703       |
| 4  | mask         | 2,763       |
| 5  | safety_shoe  | 2,006       |

---

# 3. 🎯 Intended Use

### **Primary use-cases**

- Real-time PPE compliance monitoring  
- Construction site safety analytics  
- Industrial / warehouse safety automation  
- Worker equipment detection on:
  - Helmets  
  - Safety shoes  
  - Reflective vests  
  - Gloves  
  - Masks  
  - Goggles  

### **Target Platforms**

- NVIDIA Jetson (Orin Nano, Orin NX, Xavier NX)
- ONNX Runtime
- TensorRT (FP32 / FP16 / INT8)
- NVIDIA DeepStream SDK
- Python (Ultralytics inference)

### **Not intended for**

- Legal enforcement without human review  
- Medical or clinical decisions  
- Surveillance without consent  
- Environments restricted by privacy regulations  

> ⚠️ **Human supervision is mandatory for high-risk applications.**

---

# 4. 📂 Training Data

- **Dataset source:** Custom PPE dataset (Roboflow format)  
- **Annotation format:** YOLO TXT  
- **Data types:** Indoor/outdoor industrial environments  
- **Split:**
  - Train: majority
  - Val: held-out validation
  - Test: separate set for unbiased evaluation  

Dataset is slightly imbalanced:
- **Vest** is most common  
- **safety_shoe** & **goggles** have fewer samples  

---

# 5. ⚙️ Training Procedure

- **Base model:** `yolov8n.pt` (COCO pretrained)  
- **Framework:** Ultralytics YOLOv8  
- **Hyperparameters:**
  - `epochs = 50`
  - `imgsz = 640`
  - `batch = 32`
  - `optimizer = auto`
- **Augmentations:**
  - Mosaic
  - HSV color shift
  - Random flip
  - Scaling
  - Translation  

---

# 6. 📊 Evaluation & Metrics

### **Overall Performance**

| Metric          | Value |
|-----------------|-------|
| mAP@50          | ~0.81 |
| mAP@50–95       | ~0.53 |
| Precision       | ~0.80 |
| Recall          | ~0.74 |

### **Per-Class mAP@50**

| Class        | mAP@50 |
|--------------|--------|
| Gloves       | ~0.69  |
| Vest         | ~0.90  |
| goggles      | ~0.90  |
| helmet       | ~0.90  |
| mask         | ~0.80  |
| safety_shoe  | ~0.64  |

---

# 7. ⚠️ Limitations & Ethical Considerations

### **Technical**

- Sensitive to low-light & motion blur  
- Small distant objects can reduce accuracy  
- PPE variations (colors, styles, regional differences) may affect performance

### **Bias & Data Risks**

- Dataset may not represent all ethnicities, industries, or regions globally  
- Model may fail on unseen uniform types or PPE color variations  

### **Responsible AI Guidance**

- Always include human review  
- Follow local privacy laws (GDPR, workplace regulations)  
- Use only for transparent, ethical monitoring  

---

# 8. 🚀 How to Use

## **A) Inference with Ultralytics (.pt)**

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model("image.jpg", imgsz=640)

results[0].show()  # visualize
