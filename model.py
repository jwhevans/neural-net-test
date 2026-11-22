import torch.nn as nn


class TinyCNN(nn.Module):
    """Four conv blocks + global average pool + linear head.

    28,581 parameters with the default 5 classes (the saved checkpoint holds a
    few more tensors because BatchNorm also stores running mean/var).
    """

    def __init__(self, num_classes=5):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 16),   # 64 -> 32
            block(16, 32),  # 32 -> 16
            block(32, 32),  # 16 -> 8
            block(32, 48),  # 8  -> 4
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(48, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))
