"""基于 torchvision ImageNet 预训练权重的微调模型（ResNet-18 / MobileNetV2 / ViT-B/16）。"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PretrainedClassifier(nn.Module):
    """加载预训练骨干网络，把分类头替换为 10 类后整体微调。

    预训练模型是在 224x224 图像上训练的，直接输入 32x32 的小图效果很差，
    因此在模型内部（GPU 上）先把输入双线性上采样到 input_size，
    这样数据管线仍然处理 32x32 图像，不会拖慢 CPU 端的数据加载。
    """

    def __init__(self, arch: str, num_classes: int = 10, pretrained: bool = True,
                 input_size: int = 224):
        super().__init__()
        self.input_size = input_size
        weights = "DEFAULT" if pretrained else None
        if arch == "resnet18":
            net = models.resnet18(weights=weights)
            net.fc = nn.Linear(net.fc.in_features, num_classes)
            self._cam_layer = net.layer4
        elif arch == "mobilenet_v2":
            net = models.mobilenet_v2(weights=weights)
            net.classifier[1] = nn.Linear(net.last_channel, num_classes)
            self._cam_layer = net.features[-1]
        elif arch == "vit_b_16":
            net = models.vit_b_16(weights=weights)
            net.heads.head = nn.Linear(net.heads.head.in_features, num_classes)
            self._cam_layer = None  # ViT 没有卷积特征图，本项目的 Grad-CAM 只支持 CNN
        else:
            raise ValueError(f"不支持的预训练架构: {arch}")
        self.net = net

    @property
    def cam_layer(self):
        return self._cam_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(self.input_size, self.input_size),
                          mode="bilinear", align_corners=False)
        return self.net(x)
