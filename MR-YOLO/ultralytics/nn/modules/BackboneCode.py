import torch
import torch.nn as nn
from .conv import Conv
from .conv import DWConv
import torch.nn.functional as F

class MBD2CM(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(DWR(self.c, self.c, shortcut, g, k=((3, 3), (1, 1)), e=0.5) for _ in range(n))
        # self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
class C3as(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(SIR(self.c, self.c, shortcut, g, k=((3, 3), (1, 1)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SIR(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, p=None):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * 3)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.simam = SimAM(c_)
        self.cv2 = nn.Conv2d(c_, c2, 1, 1,autopad(1, p, 1), bias=False)
        
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        y = self.cv1(x)
        # y = self.simam(y)
        return x + self.cv2(y) if self.add else self.cv2(y)


class DWR(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3),s=1, e=0.5, p=None):
        """
        c1: 输入通道数 (2C)
        c2: 输出通道数 (2C)
        shortcut: 是否使用残差连接
        e: 中间特征通道缩放比例 (默认 0.5 即 C)
        """
        super().__init__()
        c_1 = int(c1 * 1.5)  # 中间压缩通道，即 C
        c_2 = int(c2 * 0.5)
        self.c = int(c2 * 0.5)  # hidden channels
        # 初始卷积：将 2C 压缩为 C
        self.reduce = Conv(c1, c_1, k[0])
        # self.reduce = Conv(c1, c_1, k[0], act=nn.ReLU())

        # 三个不同 dilation 的 CBS 分支
        self.branch1 = DSConv(c_2, 2*c_2, k[0], d=1)
        self.branch2 = DSConv(c_2, c_2, k[0], d=3)
        self.branch3 = DSConv(c_2, c_2, k[0], d=5)

        self.simam1 = SimAM()  # 应用在每个分支后
        self.simam2 = SimAM()
        # 通道融合卷积：3C -> 2C
        self.bn1 = nn.BatchNorm2d(4 * c_2)

        self.conv = nn.Conv2d(4 * c_2, c2, 1, 1,autopad(1, p, 1), bias=False)
        self.cbam = CBAM(c2)

        # 残差连接条件
        self.add = shortcut and c1 == c2

    def forward(self, x):
        identity = x

        x = list(self.reduce(x).split((self.c, self.c, self.c), 1))
        # 分支提取
        b1 = self.branch1(x[0])       
        b2 = self.branch2(x[1])
        b3 = self.branch3(x[2])

        # b1 = self.simam1(b1)
        # b2 = self.simam2(b2)
        # b3 = self.simam2(b3)

        # 通道拼接 -> 融合 -> 输出
        x = torch.cat([b1, b2, b3], dim=1)
        x = self.bn1(x)
        x = self.conv(x)

        # x = self.cbam(x)

        # 残差连接
        return x + identity if self.add else x


class RMSPF(nn.Module):

    """"""
    def __init__(self, c1, c2,k=5):
        super(RMSPF, self).__init__()
        c_ = c1 // 2  # 中间通道
        self.cv1 = Conv(c1, c_, 1, 1)  # 通道压缩

        # 三个连续的 MPM 分支，每次输入都来自前一个 Add 的输出
        self.mpm1 = MPM(c_)
        self.mpm2 = MPM(c_)
        self.mpm3 = MPM(c_)

        self.cv2 = Conv(c_ * 4, c2, 1, 1)  # concat 后输出

    def forward(self, x):
        x1 = self.cv1(x)      # Conv1

        x2 = self.mpm1(x1) + x1  # 第一次 Add
        x3 = self.mpm2(x2)  + x1 # 第二次 Add
        x4 = self.mpm3(x3)  + x1 # 第三次 Add

        out = torch.cat([x1, x2, x3, x4], dim=1)  # Concat
        return self.cv2(out)  # Conv2




class MPM(nn.Module):
    def __init__(self, in_channels, up_kwargs={'mode': 'bilinear', 'align_corners': True}):
        super(MPM, self).__init__()
        inter_channels = int(in_channels / 4)
        # 空间池化
        self.pool1 = nn.AdaptiveAvgPool2d((3,3))
        self.pool2 = nn.AdaptiveAvgPool2d((5,5))
        # strip pooling
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))
 
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.SiLU())
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.SiLU() )
 
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(inter_channels))
 
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(inter_channels))
 
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                     nn.BatchNorm2d(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                     nn.BatchNorm2d(inter_channels))
 
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.SiLU())
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.SiLU())
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels * 2, in_channels, 1, bias=False),
                                     nn.BatchNorm2d(in_channels))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs
 
    def forward(self, x):
        b, c, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)

        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), size=[h, w], **self._up_kwargs)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), size=(h, w), **self._up_kwargs)
 
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), size=(h, w), **self._up_kwargs)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), size=(h, w), **self._up_kwargs)
        # PPM branch output
        x1 = self.conv2_5(x2_1 + x2_2 + x2_3)
        # MPM output
        x2 = self.conv2_6(x2_5 + x2_4)
 
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)
    


class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4, pos_emb=False):
        super(MHSA, self).__init__()
 
        self.heads = heads
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.pos = pos_emb
        if self.pos:
            self.rel_h_weight = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, 1, int(height)]),
                                             requires_grad=True)
            self.rel_w_weight = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, int(width), 1]),
                                             requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)
 
    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)
        # print('q shape:{},k shape:{},v shape:{}'.format(q.shape,k.shape,v.shape))  #1,4,64,256
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)  # 1,C,h*w,h*w
        # print("qkT=",content_content.shape)
        c1, c2, c3, c4 = content_content.size()
        if self.pos:
            # print("old content_content shape",content_content.shape) #1,4,256,256
            content_position = (self.rel_h_weight + self.rel_w_weight).view(1, self.heads, C // self.heads, -1).permute(
                0, 1, 3, 2)  # 1,4,1024,64
 
            content_position = torch.matmul(content_position, q)  # ([1, 4, 1024, 256])
            content_position = content_position if (
                        content_content.shape == content_position.shape) else content_position[:, :, :c3, ]
            assert (content_content.shape == content_position.shape)
            # print('new pos222-> shape:',content_position.shape)
            # print('new content222-> shape:',content_content.shape)
            energy = content_content + content_position
        else:
            energy = content_content
        attention = self.softmax(energy)
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))  # 1,4,256,64
        out = out.view(n_batch, C, width, height)
        return out





class SElayer(nn.Module):
    def __init__(self, c1, c2, ratio=16):
        super(SElayer, self).__init__()
        #c*1*1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)



"""
通道注意力模型: 通道维度不变，压缩空间维度。该模块关注输入图片中有意义的信息。
1）假设输入的数据大小是(b,c,w,h)
2）通过自适应平均池化使得输出的大小变为(b,c,1,1)
3）通过2d卷积和sigmod激活函数后，大小是(b,c,1,1)
4）将上一步输出的结果和输入的数据相乘，输出数据大小是(b,c,w,h)。
"""
class ChannelAttention(nn.Module):
    # Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))

"""
空间注意力模块：空间维度不变，压缩通道维度。该模块关注的是目标的位置信息。
1） 假设输入的数据x是(b,c,w,h)，并进行两路处理。
2）其中一路在通道维度上进行求平均值，得到的大小是(b,1,w,h)；另外一路也在通道维度上进行求最大值，得到的大小是(b,1,w,h)。
3） 然后对上述步骤的两路输出进行连接，输出的大小是(b,2,w,h)
4）经过一个二维卷积网络，把输出通道变为1，输出大小是(b,1,w,h)
4）将上一步输出的结果和输入的数据x相乘，最终输出数据大小是(b,c,w,h)。
"""
class SpatialAttention(nn.Module):
    # Spatial-attention module
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))

class CBAM(nn.Module):
    # Convolutional Block Attention Module
    def __init__(self, c1, kernel_size=7):  # ch_in, kernels
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)
        
        

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))



"""
CA注意力机制模块
"""
 
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
 
    def forward(self, x):
        return self.relu(x + 3) / 6
 
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
 
    def forward(self, x):
        return x * self.sigmoid(x)
 
class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        oup = inp
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
 
        mip = max(8, inp // reduction)
 
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
 
    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
 
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
 
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
 
        out = identity * a_w * a_h
 
        return out
    

    
class SimAM(nn.Module):
    def __init__(self, channels=None, out_channels=None, e_lambda=1e-4, use_leaky_relu=False, learnable_bias=False):
        super(SimAM, self).__init__()

        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda
        self.use_leaky_relu = use_leaky_relu
        self.learnable_bias = learnable_bias
        
        if self.use_leaky_relu:
            self.activation = nn.LeakyReLU(negative_slope=0.1)

        # 可学习的偏置
        if self.learnable_bias:
            self.bias = nn.Parameter(torch.zeros(1))  # 可学习的偏置项

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f, use_leaky_relu=%s)' % (self.e_lambda, str(self.use_leaky_relu)))
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        
        # 引入标准差来平衡均值和方差
        std_dev = torch.std(x, dim=[2, 3], keepdim=True)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        
        # 可学习的偏置调整
        if self.learnable_bias:
            y += self.bias

        return x * self.activation(y)


      
class DSConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, d=1, act=True):
        """
        Depthwise Separable Convolution
        
        Args:
            c1: 输入通道数
            c2: 输出通道数
            k: 卷积核大小 (默认为3)
            s: 步幅 (默认为1)
            p: 填充 (默认为None)
            d: 膨胀 (默认为1)
            act: 是否使用激活函数，默认使用SiLU
        """
        super().__init__()

        # Depthwise Convolution: 每个卷积核处理一个通道
        self.depthwise_conv = nn.Conv2d(c1, c1, k, s, autopad(3, p, d), groups=c1, dilation=d, bias=False)
        # self.depthwise_conv = nn.Conv2d(c1, c2, k, s, autopad(3, p, d), groups=c1, dilation=d, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)

        # Pointwise Convolution: 1x1卷积用于跨通道的特征融合
        self.pointwise_conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)

        # 激活函数
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        """
        Forward pass for Depthwise Separable Convolution.
        
        Args:
            x: 输入数据
        
        Returns:
            输出数据
        """
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.pointwise_conv(x)
        x = self.bn2(x)
        x = self.act(x)

        return x

def autopad(k, p=None, d=1):
    """
    自动填充功能
    k: 卷积核大小
    p: 填充大小
    d: 膨胀
    """
    if p is None:
        p = (k - 1) // 2 * d  # "same" padding
    return p
