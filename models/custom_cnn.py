import torch
import torch.nn as nn

# A compact custom CNN for regression (health index)
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pdrop=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(pdrop)
        )

    def forward(self, x):
        return self.block(x)

class CustomCNN(nn.Module):
    def __init__(self, in_channels=3, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, pdrop=dropout),
            ConvBlock(32, 64, pdrop=dropout),
            ConvBlock(64, 128, pdrop=dropout),
            ConvBlock(128, 256, pdrop=dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()  # output in [0,1]
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)
