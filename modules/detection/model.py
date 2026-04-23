# modules/detection/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Reusable convolutional block:
        Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU → MaxPool2d → Dropout2d

    Constructor args:
        in_channels  (int):   input channels
        out_channels (int):   output channels for BOTH Conv layers
        dropout_p    (float): Dropout2d probability, default 0.25

    bias=False in all Conv2d because BatchNorm follows immediately.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop  = nn.Dropout2d(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop(x)
        return x


class MalTwinCNN(nn.Module):
    """
    Three-block CNN for grayscale malware image classification.

    Input:  (batch_size, 1, 128, 128)  — single-channel grayscale
    Output: (batch_size, num_classes)  — raw logits (NO softmax)

    Architecture:
        Input (1, 128, 128)
            ↓
        block1: ConvBlock(1 → 32)      → (32, 64, 64)   after MaxPool
            ↓
        block2: ConvBlock(32 → 64)     → (64, 32, 32)   after MaxPool
            ↓
        block3: ConvBlock(64 → 128)    → (128, 16, 16)  after MaxPool
            ↓                            ← self.gradcam_layer = self.block3.conv2
        pool:   AdaptiveAvgPool2d(4,4) → (128, 4, 4)
            ↓
        flatten                        → (2048,)
            ↓
        classifier:
            Linear(2048 → 512)
            ReLU
            Dropout(p=0.5)
            Linear(512 → num_classes)
            ↓
        raw logits (num_classes,)

    CRITICAL:
        self.gradcam_layer = self.block3.conv2
        This MUST be set — it is tested explicitly.
        It is used by Module 7 (Grad-CAM) to register backward hooks.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.block1 = ConvBlock(1, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.pool   = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

        # Grad-CAM hook target — MUST be the second conv of block3
        self.gradcam_layer = self.block3.conv2

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Kaiming normal for Conv2d, constant init for BatchNorm, Xavier for Linear.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x  # raw logits — NO softmax here

