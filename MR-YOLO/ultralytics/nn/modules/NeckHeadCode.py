import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv

import math

# ------------------------------
# Overall Module
# ------------------------------
class MSFFM(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, c1,c2, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        c_ = int(c1* 1)

        self.d = dimension
        self.local_branch = LocalBranch(c_)
        self.global_branch = GlobalBranch(c_)
        self.weight_net = WeightNet(c_)

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""

        tmp = torch.cat(x, self.d)
        outl = self.local_branch(tmp)
        outg = self.global_branch(tmp)
        wl, wg = self.weight_net(tmp)
        out = outl * wl + outg * wg


        return out
    
class DSConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)

# ------------------------------
# Local Branch Module
# ------------------------------
class LocalBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = DSConv(channels, channels)


    def forward(self, x):
        return self.block(x)

# ------------------------------
# Global Branch Module
# ------------------------------
class GlobalBranch(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # [B, C, 1, 1]
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)       # [B, 1, C]
        y = self.conv1d(y)                        # [B, 1, C]
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y



# ------------------------------
# WeightNet Module
# ------------------------------
class WeightNet(nn.Module):
    def __init__(self, channels, n_branches=2, reduction=4):
        super().__init__()
        self.n_branches = 2
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, n_branches, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.pool(x)  # [B, C, 1, 1]
        weights = self.fc(x)  # [B, n_branches, 1, 1]
        return torch.chunk(weights, self.n_branches, dim=1)

