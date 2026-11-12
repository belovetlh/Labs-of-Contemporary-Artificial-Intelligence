"""模型注册表：按名称构建模型。"""
import torch.nn as nn

from .pretrained import PretrainedClassifier
from .resnet import ResNet
from .simple_cnn import BaselineCNN

# 名称以 _pt 结尾的是 ImageNet 预训练模型，其余均从零训练
MODEL_NAMES = ("simple_cnn", "resnet18", "resnet18_se", "resnet18_cbam",
               "resnet18_pt", "mobilenet_v2_pt", "vit_b_16_pt")


def is_pretrained(name: str) -> bool:
    return name.endswith("_pt")


def build_model(name: str, num_classes: int = 10, dropout: float = 0.0,
                load_weights: bool = True) -> nn.Module:
    """构建模型。

    dropout       分类头前的 Dropout 概率（仅对从零训练的模型生效）
    load_weights  预训练模型是否下载 ImageNet 权重；离线检查代码时可设为 False
    """
    if name == "simple_cnn":
        return BaselineCNN(num_classes, dropout=dropout)
    if name == "resnet18":
        return ResNet(num_classes=num_classes, attention="none", dropout=dropout)
    if name == "resnet18_se":
        return ResNet(num_classes=num_classes, attention="se", dropout=dropout)
    if name == "resnet18_cbam":
        return ResNet(num_classes=num_classes, attention="cbam", dropout=dropout)
    if is_pretrained(name):
        return PretrainedClassifier(name[:-len("_pt")], num_classes, pretrained=load_weights)
    raise ValueError(f"未知的模型: {name}，可选 {MODEL_NAMES}")
