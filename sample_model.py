"""
sample_model.py
---------------
Example custom PyTorch model file for the PyTorch DAG Optimizer.

Upload this file to the web app to see a custom model analysed, optimized,
and exported as both a DAG JSON and a TorchScript .pt file.

Rules:
  1. Define `get_model()` → returns an nn.Module
  2. Optionally define MODEL_NAME (str) and INPUT_SHAPE (list[int])
"""

import torch.nn as nn

# ── Optional metadata ──────────────────────────────────────────────────────
MODEL_NAME  = "MiniCNN"
INPUT_SHAPE = [1, 3, 32, 32]   # batch=1, channels=3, height=32, width=32


# ── Model definition ────────────────────────────────────────────────────────

class MiniCNN(nn.Module):
    """
    A small CNN with Conv-BN-ReLU blocks — ideal for demonstrating
    the optimizer's Conv-BN fusion and dead-node elimination passes.
    """

    def __init__(self):
        super().__init__()

        # Block 1: Conv-BN-ReLU (will be fused into a single node)
        self.conv1 = nn.Conv2d(3,  32, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        # Block 2: Conv-BN-ReLU (will be fused)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        # Block 3: Conv-BN only (Conv-BN folding pass will merge weights)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(64)

        # Classifier head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))   # CBR block 1
        x = self.relu2(self.bn2(self.conv2(x)))   # CBR block 2
        x = self.bn3(self.conv3(x))               # CB block (no ReLU)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


def get_model() -> nn.Module:
    """Required entry point — must return an nn.Module."""
    return MiniCNN()
