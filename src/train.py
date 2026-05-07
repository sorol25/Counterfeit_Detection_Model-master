# src/train.py

import os
import torch
from ultralytics import YOLO

# -------------------------------
# Configuration
# -------------------------------

DATA_YAML = os.path.abspath("datasets/counterfeit_med_detection/data.yaml")

# You can switch to yolov8s.pt or yolov8m.pt for better accuracy
MODEL_NAME = "yolov8n.pt"

# Directory to save the trained model
SAVE_DIR = os.path.abspath("models")

# -------------------------------
# Training Hyperparameters
# -------------------------------

EPOCHS = 10
BATCH_SIZE = 8
IMG_SIZE = 1280

# -------------------------------
# Training Function
# -------------------------------

def train_model():

    print("🚀 Training started on counterfeit_med_detection dataset...")

    # Check dataset yaml
    if not os.path.exists(DATA_YAML):
        print(f"❌ Dataset YAML not found: {DATA_YAML}")
        return

    # Create save directory if missing
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Detect device
    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"🖥️ Using device: {device}")

    # Load YOLO model
    model = YOLO(MODEL_NAME)

    # Start training
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        project=SAVE_DIR,
        name="counterfeit_capsule_model",
        pretrained=True,
        device=device,
        cache=False,
        workers=0,
        verbose=True
    )

    print(f"\n✅ Training complete! Model saved in: {SAVE_DIR}/counterfeit_capsule_model")

# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":
    train_model()