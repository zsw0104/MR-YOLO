import torch
import torch.nn as nn
import torchvision
from .block import PSABlock, C2PSA
from .conv import Conv

class DeformConv(nn.Module):

    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()

        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out


class deformable_LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = DeformConv(dim, kernel_size=(5, 5), padding=2, groups=dim)
        self.conv_spatial = DeformConv(dim, kernel_size=(7, 7), stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class deformable_LKA_Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = deformable_LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class PSABlock_LKA(PSABlock):

    def __init__(self, c, qk_dim=16, pdim=32, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__(c)

        self.ffn = deformable_LKA_Attention(c)


class C2PSA_LKA(C2PSA):

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__(c1, c2)
        assert c1 == c2
        self.c = int(c1 * e)

        self.m = nn.Sequential(*(PSABlock_LKA(self.c, qk_dim=16, pdim=32) for _ in range(n)))

class C2DLKA(nn.Module):
    """Standard bottleneck."""
    #"仿照c2f模块程序"
    def __init__(self, c1, c2, e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        self.c = int(c2 * 0.5)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(self.c, self.c, 1, 1)
        self.cv3 = Conv(c2, c2, 1, 1)
        self.dlka = C2PSA_LKA(self.c, self.c)


    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        y = self.cv1(x).chunk(2, 1)
        y0 = y[0]
        y1 = y[1]
        z0 = self.cv2(self.dlka(self.cv2(y0)))
        z1 = self.cv2(self.dlka(self.cv2(y1)))
        f0 = y0 + z0
        f1 = y1 + z1
        k = torch.cat((f0, f1), 1)
        return self.cv3(k)
