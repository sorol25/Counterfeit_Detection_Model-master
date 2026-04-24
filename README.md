# 🧠 Capsule Counterfeit Detection using YOLOv8

A deep learning-based computer vision system for detecting and classifying **genuine vs counterfeit medicinal capsules** using state-of-the-art object detection (YOLOv8).

This project is designed for **real-world pharmaceutical authentication scenarios**, enabling automated visual verification of capsule authenticity across multiple brands.

---

## 🚀 Project Highlights

* 🔍 Multi-class capsule detection (8 classes: authentic + counterfeit)
* ⚡ Real-time capable using YOLOv8 lightweight models
* 🧠 Trained on curated Roboflow dataset
* 📦 Modular and production-ready codebase
* 🔁 Easily extendable for new brands and capsules
* 📊 Evaluation pipeline included

---

## 📁 Project Structure

```
capsule-counterfeit-detection/
│
├── datasets/                  # Training and validation datasets
├── models/                    # Trained model weights
│   ├── counterfeit_capsule_model/
│   └── verification_model/
│
├── runs/                      # YOLO training outputs (logs, metrics)
├── results/                   # Evaluation results
│
├── src/                       # Core source code
│   ├── train.py
│   ├── train_verification.py
│   ├── evaluate.py
│   ├── evaluate_verification.py
│   ├── preprocess.py
│   └── preprocess_verification.py
│
├── test/                      # Test scripts & inference samples
├── utils/                     # Helper utilities
│
├── verification_dataset/
├── verification_dataset_test/
│
├── main.py                    # Inference entry point
├── requirements.txt
├── yolov8m.pt                 # Pretrained YOLOv8 model
├── yolov10n.pt                # Experimental model
└── README.md
```

---

## 🧠 Model Overview

* **Architecture**: YOLOv8 (Ultralytics)
* **Task Type**: Object Detection + Multi-class Classification
* **Input**: RGB images of medicinal capsules
* **Output**: Bounding boxes with class labels

### 🎯 Detected Classes (8 Total)

* authentic_BrandW
* authentic_BrandX
* authentic_BrandY
* authentic_BrandZ
* counterfeit_BrandW
* counterfeit_BrandX
* counterfeit_BrandY
* counterfeit_BrandZ

---

## 📊 Dataset Information

* **Format**: YOLOv8 annotation format
* **Source**: Roboflow Universe
* **Dataset Link**: [https://universe.roboflow.com/medetect/medetect-9kphx/dataset/11](https://universe.roboflow.com/medetect/medetect-9kphx/dataset/11)
* **License**: CC BY 4.0

---

## ⚙️ Configuration (`data.yaml`)

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 8
names: [
  'authentic_BrandW', 'authentic_BrandX',
  'authentic_BrandY', 'authentic_BrandZ',
  'counterfeit_BrandW', 'counterfeit_BrandX',
  'counterfeit_BrandY', 'counterfeit_BrandZ'
]
```

---

## 🛠️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/sorol25/Counterfeit_Detection_Model-master.git
cd Counterfeit_Detection_Model-master
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 🧠 Train Model

```bash
python src/train.py
```

### 📊 Evaluate Model

```bash
python src/evaluate.py
```

### 🔍 Run Inference

```bash
python main.py
```

---

## 📈 Features

* ✅ Multi-brand counterfeit detection system
* ⚡ Optimized YOLOv8 inference pipeline
* 📦 Clean modular architecture
* 🔁 Supports retraining and fine-tuning
* 🧪 Evaluation pipeline for performance tracking
* 🚀 Ready for deployment integration

---

## 🧪 Future Improvements

* 🧠 Siamese / Triplet Loss-based verification model (few-shot learning)
* 🔥 Grad-CAM visualization for explainability
* 🌐 Web deployment using Flask / FastAPI / Streamlit
* 📱 Mobile deployment (ONNX / TensorRT optimization)
* 📊 Expanded dataset with more pharmaceutical brands
* 🛰️ Real-time camera integration for live verification

---

## 🤝 Contributing

Contributions are welcome and appreciated.

If you would like to contribute:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

For major changes, please open an issue first to discuss the approach.

---

## 📄 License

This project is licensed under the **MIT License** (or update if different).

---

## 👨‍💻 Author

**Yeamine Alam Sorol**

* GitHub: [@sorol25](https://github.com/sorol25)
* LinkedIn: www.linkedin.com/in/yeamine-alam-sorol-746831347

---

## ⭐ Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* Roboflow Dataset Platform
* Open-source computer vision community

---


