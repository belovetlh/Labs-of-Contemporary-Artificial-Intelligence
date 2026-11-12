"""注意力模块：SE（通道注意力）与 CBAM（通道 + 空间注意力）。"""
import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (Hu et al., 2018)。

    Squeeze：全局平均池化把每个通道压缩成一个标量；
    Excitation：两层全连接学习各通道的重要性权重，再逐通道缩放特征图。
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class ChannelAttention(nn.Module):
    """CBAM 的通道注意力：平均池化与最大池化两路共享同一个 MLP，结果相加后过 Sigmoid。"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(x.mean(dim=(2, 3), keepdim=True))
        mx = self.mlp(x.amax(dim=(2, 3), keepdim=True))
        return torch.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    """CBAM 的空间注意力：沿通道维做平均/最大池化，拼接后用 7x7 卷积得到空间权重图。"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., 2018)：先通道注意力，再空间注意力。"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel(x)
        return x * self.spatial(x)


def build_attention(name: str, channels: int) -> nn.Module:
    """按名称创建注意力模块；'none' 返回恒等映射。"""
    if name == "none":
        return nn.Identity()
    if name == "se":
        return SEBlock(channels)
    if name == "cbam":
        return CBAM(channels)
    raise ValueError(f"未知的注意力类型: {name}")
