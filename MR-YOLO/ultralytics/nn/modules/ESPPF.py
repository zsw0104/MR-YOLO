import torch
import torch.nn as nn
from .conv import Conv

class ESPPF(nn.Module):


    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        c__ = c_//2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c__ * 4, c_, 1, 1)
        self.cv3 = Conv(c_, c__, 1, 1)
        self.cv4 = Conv(c_*2, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        z = self.cv1(x)
        y = [self.cv3(z)]
        y.extend(self.m(y[-1]) for _ in range(3))
        k = self.cv2(torch.cat(y, 1))
        t = torch.cat((z, k), 1)
        return self.cv4(t)
