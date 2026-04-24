# models/capsule_verification_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class CapsuleVerificationModel(nn.Module):
    def __init__(self, embedding_size=512):
        super(CapsuleVerificationModel, self).__init__()

        # Load pretrained EfficientNet backbone
        backbone = EfficientNet.from_pretrained('efficientnet-b3')

        # Remove classification head
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Fully connected layer for embedding projection
        self.fc = nn.Linear(self.backbone[-1].in_features, embedding_size)

    def forward(self, x):
        # Feature extraction
        x = self.backbone(x)

        # Flatten features
        x = x.view(x.size(0), -1)

        # Embedding projection
        x = self.fc(x)

        # L2 normalization for verification embeddings
        x = F.normalize(x, p=2, dim=1)

        return x