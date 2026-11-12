"""损失函数：交叉熵（可带标签平滑）与 Focal Loss。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017)：FL = -(1 - p_t)^gamma * log(p_t)。

    通过 (1 - p_t)^gamma 降低已分对的简单样本的权重，让训练更关注难样本；
    gamma = 0 时退化为普通交叉熵。
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 混合精度下 logits 可能是 float16，转成 float32 再算对数概率，避免数值不稳定
        log_p = F.log_softmax(logits.float(), dim=1)
        log_pt = log_p.gather(1, target.unsqueeze(1)).squeeze(1)
        return (-(1.0 - log_pt.exp()) ** self.gamma * log_pt).mean()


def build_criterion(name: str, label_smoothing: float = 0.0, focal_gamma: float = 2.0) -> nn.Module:
    """按名称创建训练损失。

    ce     交叉熵；label_smoothing > 0 时把 one-hot 目标软化为
           (1 - eps) * one_hot + eps / K，抑制模型过度自信
    focal  Focal Loss
    """
    if name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if name == "focal":
        return FocalLoss(focal_gamma)
    raise ValueError(f"未知的损失函数: {name}")
