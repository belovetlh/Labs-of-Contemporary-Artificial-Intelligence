"""自行设计的基线 CNN：VGG 风格的三阶段卷积网络，带 BatchNorm 与 Dropout。"""
import torch
import torch.nn as nn


def conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    """3x3 卷积 + BatchNorm + ReLU。卷积后接 BN，因此卷积层不需要偏置。"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class BaselineCNN(nn.Module):
    """基线 CNN。

    结构：3 个阶段，每个阶段 2 个 [Conv3x3-BN-ReLU]，通道数 64 -> 128 -> 256；
    前两个阶段后接 2x2 最大池化（32 -> 16 -> 8），最后用全局平均池化代替大的全连接层，
    再经 Dropout 和一个线性层输出 10 类 logits。约 115 万参数。
    """

    def __init__(self, num_classes: int = 10, widths=(64, 128, 256), dropout: float = 0.3):
        super().__init__()
        self.stage1 = nn.Sequential(conv_bn_relu(3, widths[0]), conv_bn_relu(widths[0], widths[0]))
        self.stage2 = nn.Sequential(conv_bn_relu(widths[0], widths[1]), conv_bn_relu(widths[1], widths[1]))
        self.stage3 = nn.Sequential(conv_bn_relu(widths[1], widths[2]), conv_bn_relu(widths[2], widths[2]))
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(widths[2], num_classes),
        )

    @property
    def cam_layer(self) -> nn.Module:
        """Grad-CAM 使用的目标层：最后一个卷积阶段（8x8 特征图）。"""
        return self.stage3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.stage1(x))
        x = self.pool(self.stage2(x))
        x = self.stage3(x)
        return self.head(x)
