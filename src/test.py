import cv2
from pathlib import Path

# Define dataset paths
img_path = Path("datasets/counterfeit_med_detection/train/images")
label_path = Path("datasets/counterfeit_med_detection/train/labels")

# Pick the first label file
sample = list(label_path.glob("*.txt"))[0]

# Try supported image extensions without changing logic
img_file = None
for ext in [".jpg", ".jpeg", ".png"]:
    temp_file = img_path / f"{sample.stem}{ext}"
    if temp_file.exists():
        img_file = temp_file
        break

print("Sample Label File:", sample)
print("Image Exists:", img_file.exists() if img_file else False)

# Read and display label content
with open(sample, "r") as f:
    print("Label File Content:")
    print(f.read())

# Load image safely
img = cv2.imread(str(img_file)) if img_file else None

print("Image shape:", img.shape if img is not None else "Image not loaded")