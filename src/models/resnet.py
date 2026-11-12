"""适配 CIFAR 32x32 输入的 ResNet-18，可在每个残差块中插入注意力模块。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import build_attention


class BasicBlock(nn.Module):
    """残差基本块：两个 3x3 卷积 + 恒等/投影捷径。

    注意力模块作用在残差分支的输出上（与 SE-ResNet 原论文一致），然后再与捷径相加。
    """

    def __init__(self, in_planes: int, planes: int, stride: int = 1, attention: str = "none"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.attention = build_attention(attention, planes)
        # 尺寸或通道数变化时，用 1x1 卷积把捷径投影到相同形状
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.attention(self.bn2(self.conv2(out)))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    """CIFAR 版 ResNet。

    与 ImageNet 版的区别：首层用 3x3 卷积且不做下采样、去掉最大池化，
    避免 32x32 的小图在网络前端就丢失过多空间信息。
    四个阶段的通道数为 64/128/256/512，特征图尺寸为 32/16/8/4。
    """

    def __init__(self, num_blocks=(2, 2, 2, 2), num_classes: int = 10,
                 attention: str = "none", dropout: float = 0.0):
        super().__init__()
        self.in_planes = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, num_blocks[0], 1, attention)
        self.layer2 = self._make_layer(128, num_blocks[1], 2, attention)
        self.layer3 = self._make_layer(256, num_blocks[2], 2, attention)
        self.layer4 = self._make_layer(512, num_blocks[3], 2, attention)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int, attention: str) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, s, attention))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """卷积层用 Kaiming 初始化（适配 ReLU），BN 初始化为恒等变换。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def cam_layer(self) -> nn.Module:
        """Grad-CAM 使用的目标层：最后一个残差阶段（4x4 特征图）。"""
        return self.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.head(x)
