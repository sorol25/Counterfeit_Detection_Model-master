# src/dataset.py

import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class CapsuleDataset(Dataset):
    def __init__(self, split="train", root_dir="datasets/counterfeit_med_detection"):
        self.split = split
        self.root_dir = root_dir

        # Image directory path
        self.image_dir = os.path.join(root_dir, split, "images")

        # Label mapping
        self.label_map = self._create_label_map()

        # Collect image paths
        self.image_paths = [
            os.path.join(self.image_dir, fname)
            for fname in os.listdir(self.image_dir)
            if fname.endswith((".jpg", ".png"))
        ]

        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def _create_label_map(self):
        return {
            "authentic_BrandW": 0,
            "authentic_BrandX": 1,
            "authentic_BrandY": 2,
            "authentic_BrandZ": 3,
            "counterfeit_BrandW": 4,
            "counterfeit_BrandX": 5,
            "counterfeit_BrandY": 6,
            "counterfeit_BrandZ": 7
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Extract label from filename prefix
        label_name = os.path.basename(img_path).split("_")[0]

        # Map label to integer class
        label = self.label_map[label_name]

        # Apply transforms
        image = self.transform(image)

        return image, label